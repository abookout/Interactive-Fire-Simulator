using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;

public class Main : MonoBehaviour
{
    // This should match with the SPH CS!
    const int ThreadGroupSize = 64;
    const int MaxNumParticles = 256;

    // Buffer and parameter IDs
    static readonly int
        dataInputBufID = Shader.PropertyToID("_DataInputBuffer"),
        accelerationInputBufID = Shader.PropertyToID("_AccelerationInputBuffer"),

        dataOutputBufID = Shader.PropertyToID("_DataOutputBuffer"),
        accelerationOutputBufID = Shader.PropertyToID("_AccelerationOutputBuffer"),

        deltaTimeID = Shader.PropertyToID("_DeltaTime"),
        numParticlesID = Shader.PropertyToID("_NumParticles"),
        particleMassID = Shader.PropertyToID("_ParticleMass"),
        pressureStiffnessID = Shader.PropertyToID("_PressureStiffness"),
        referenceDensityID = Shader.PropertyToID("_ReferenceDensity"),
        extAccelerationsID = Shader.PropertyToID("_ExternalAccelerations");

    [Header("Set in inspector")]
    [SerializeField] new ParticleSystem particleSystem;
    [SerializeField, Range(1, MaxNumParticles)] int numParticles = 32;
    [SerializeField] float particleMass = 0.1f;
    [SerializeField] float pressureStiffness = 1f;
    [SerializeField] float referenceDensity = 1f;      // Probably choose atmostpheric pressure
    [SerializeField] Vector3 externalAccelerations = new(0, -9.8f, 0);      // Just gravity, for now at least

    // Main compute shader for calulating update to particle accelerations
    [Header("Buffer & shader object properties")]
    [SerializeField] ComputeShader SPHComputeShader;

    // For testing
    [SerializeField] RenderTexture tex;

    struct ParticleData
    {
        public Vector3 position;
        public Vector3 velocity;
    }
    const int sizeofParticleData = sizeof(float) * 3 * 2;

    // Particle positions and current velocities are given to the acceleration compute shader as an input.
    //  Position is then set by the ComputePosition kernel by integration.
    [SerializeField] ComputeBuffer particleDataInputBuffer;
    [SerializeField] ComputeBuffer particleDataOutputBuffer;
    // Buffer for the calculation results
    [SerializeField] ComputeBuffer particleAccelerationBuffer;

    // Using OnEnable instead of start or awake so that the buffers are refreshed every hot reload
    private void OnEnable()
    {
        Unity.Collections.LowLevel.Unsafe.UnsafeUtility.SetLeakDetectionMode(NativeLeakDetectionMode.EnabledWithStackTrace);

        InitializeBuffers();
    }
    private void OnDisable()
    {
        ReleaseBuffers();
    }
    private void Update()
    {
        // Make sure the number of particles hasn't changed
        if (particleSystem.particleCount != numParticles)
        {
            // It has changed, so need to either destroy some or instantiate more
            if (particleSystem.particleCount > numParticles)
            {
                // Destroy some particles to bring it back down to numParticles
                ParticleSystem.Particle[] particles = new ParticleSystem.Particle[MaxNumParticles];
                particleSystem.GetParticles(particles);
                // Only set numParticles
                particleSystem.SetParticles(particles, numParticles);
            } 
            else
            {
                // particleSystem.particleCount < numParticles, so emit the difference
                particleSystem.Emit(numParticles - particleSystem.particleCount);
            }
        }
        
        // Send positions and velocities to GPU
        WriteParticleDataToGPU();

        // Perform SPH calculations
        SimulateParticles();
    }

    void InitializeBuffers()
    {
        if (tex == null)
        {
            tex = new RenderTexture(numParticles, numParticles, 0, RenderTextureFormat.ARGBFloat);
            tex.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            tex.volumeDepth = numParticles;
            tex.enableRandomWrite = true;
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Point;
            tex.Create();
        }

        int sizeofVec3 = sizeof(float) * 3;

        // note: SubUpdates prevents RenderDoc from viewing the buffer contents

        // Allocate the maximum amount of particles so that we don't need to keep creating new buffers when the number of particles changes
        particleDataInputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleData);
        particleDataOutputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleData);
        particleAccelerationBuffer = new ComputeBuffer(MaxNumParticles, sizeofVec3);

        // Initialize buffers, using unity's particle system emission shape to choose random initial positions in a box
        //var psShape = particleSystem.shape;
        particleSystem.Emit(numParticles);
        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[numParticles];
        particleSystem.GetParticles(particles);

        ParticleData[] dataArr = new ParticleData[MaxNumParticles];
        Vector3[] accArr = new Vector3[MaxNumParticles];
        for (int i = 0; i < MaxNumParticles; i++)
        {
            if (i < numParticles)
                dataArr[i].position = particles[i].position;
            else
                dataArr[i].position = Vector3.zero;
            dataArr[i].velocity = Vector3.zero;
            accArr[i] = Vector3.zero;
        }
        particleDataInputBuffer.SetData(dataArr);
        particleDataOutputBuffer.SetData(dataArr);       // Initialize output with the same data as input
        particleAccelerationBuffer.SetData(accArr);
    }

    void ReleaseBuffers()
    {
        particleDataInputBuffer.Release();
        particleDataOutputBuffer.Release();
        particleAccelerationBuffer.Release();
    }

    // Write current particle positions and velocities for acceleration CS
    void WriteParticleDataToGPU()
    {
        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[numParticles];
        particleSystem.GetParticles(particles);

        //TODO: maintain position and velocity arrays so don't need to access from particle system? Not sure if this is a problem
        ParticleData[] dataArray = new ParticleData[numParticles];

        for (int i = 0; i < numParticles; i++)
        {
            dataArray[i].position = particles[i].position;
            dataArray[i].velocity = particles[i].velocity;
        }

        particleDataInputBuffer.SetData(dataArray);

        // This may speed it up later, using SubUpdates 

        //NativeArray<Vector3> GPUPosArray = particlePositionBuffer.BeginWrite<Vector3>(0, numParticles);
        //NativeArray<Vector3> GPUVelArray = particleVelocityBuffer.BeginWrite<Vector3>(0, numParticles);
        //for (int i = 0; i < numParticles; i++)
        //{
        //    GPUPosArray[i] = particles[i].position;
        //    GPUVelArray[i] = particles[i].velocity;
        //}
        //particleVelocityBuffer.EndWrite<Vector3>(numParticles);
        //particlePositionBuffer.EndWrite<Vector3>(numParticles);
    }

    // Dispatch a job to the GPU to run a new SPH simulation step
    void SimulateParticles()
    {
        int kernelID = SPHComputeShader.FindKernel("ComputeAcceleration");

        // Acceleration computation uses position and velocity as inputs, and accel as output
        SPHComputeShader.SetBuffer(kernelID, dataInputBufID, particleDataInputBuffer);
        //SPHComputeShader.SetBuffer(kernelID, velocityInputBufID, particleVelocityBuffer);
        SPHComputeShader.SetBuffer(kernelID, accelerationOutputBufID, particleAccelerationBuffer);
        // Set debug property because it blows up if you don't
        SPHComputeShader.SetTexture(kernelID, Shader.PropertyToID("Result"), tex);

        SPHComputeShader.SetInt(numParticlesID, numParticles);
        SPHComputeShader.SetFloat(particleMassID, particleMass);
        SPHComputeShader.SetFloat(pressureStiffnessID, pressureStiffness);
        SPHComputeShader.SetFloat(referenceDensityID, referenceDensity);
        SPHComputeShader.SetVector(extAccelerationsID, externalAccelerations);


        // Squared num particles because of the size of the test texture
        //  Make sure there's at least one thread group!
        int numThreadGroups = Mathf.Max(1, numParticles * numParticles * numParticles / ThreadGroupSize);

        if (numThreadGroups > 65535)
        {
            Debug.LogWarning("Number of thread groups (" + numThreadGroups + ") would exceed the maximum of 65535.");
            numThreadGroups = 65535;
        }

        SPHComputeShader.Dispatch(kernelID, numThreadGroups, 1, 1);


        // Dispatch the position computation CS
        kernelID = SPHComputeShader.FindKernel("ComputePosition");

        // Integration uses position, velocity, and acceleration as inputs, and outputs new position and velocity
        SPHComputeShader.SetBuffer(kernelID, dataInputBufID, particleDataInputBuffer);
        //SPHComputeShader.SetBuffer(kernelID, velocityInputBufID, particleVelocityBuffer);
        SPHComputeShader.SetBuffer(kernelID, accelerationInputBufID, particleAccelerationBuffer);

        SPHComputeShader.SetBuffer(kernelID, dataOutputBufID, particleDataOutputBuffer);

        SPHComputeShader.SetFloat(deltaTimeID, Time.deltaTime);
        SPHComputeShader.Dispatch(kernelID, numThreadGroups, 1, 1);

        // Positions buffer is now updated, set particle positions.
        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[numParticles];
        particleSystem.GetParticles(particles);

        ParticleData[] dataArray = new ParticleData[numParticles];
        particleDataOutputBuffer.GetData(dataArray);

        for (int i = 0; i < numParticles; i++)
        {
            particles[i].position = dataArray[i].position;
            particles[i].velocity = dataArray[i].velocity;
        }
        particleSystem.SetParticles(particles);
    }
}


// This works with [numthreads(8, 8, 1)]
//SPHComputeShader.Dispatch(0, tex.width / 8, tex.height / 8, 1);

// This works with [numthreads(64, 1, 1)]
//SPHComputeShader.Dispatch(0, tex.width*4, 1, 1);

// To process x items, with a kernel group size of ThreadGroupSize*1*1, call Dispatch(x/ThreadGroupSize, 1, 1)!!
//SPHComputeShader.Dispatch(0, 256*256 / ThreadGroupSize, 1, 1);
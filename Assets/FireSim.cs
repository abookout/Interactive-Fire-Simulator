using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;

public class Main : MonoBehaviour
{
    // This should match with the SPH CS!
    const int ThreadGroupSize = 64;
    const int MaxNumParticles = 2048;

    // Buffer and parameter IDs
    static readonly int
    // Buffers
        dataInputBufID = Shader.PropertyToID("_DataInputBuffer"),

        dataOutputBufID = Shader.PropertyToID("_DataOutputBuffer"),

        // Parameters
        deltaTimeID = Shader.PropertyToID("_DeltaTime"),
        numParticlesID = Shader.PropertyToID("_NumParticles"),
        particleMassID = Shader.PropertyToID("_ParticleMass"),
        pressureStiffnessID = Shader.PropertyToID("_PressureStiffness"),
        viscosityID = Shader.PropertyToID("_Viscosity"),
        referenceDensityID = Shader.PropertyToID("_ReferenceDensity"),
        extAccelerationsID = Shader.PropertyToID("_ExternalAccelerations");

    [Header("Set in inspector")]
    [SerializeField] new ParticleSystem particleSystem;
    [SerializeField, Range(1, MaxNumParticles)] int numParticles = 32;
    [SerializeField] float particleMass = 0.1f;
    [SerializeField] float viscosity = 15.8f;        // Kinematic viscosity of air is 15.8 m^2/s (text p.276)
    [SerializeField] float pressureStiffness = 1f;
    [SerializeField] float referenceDensity = 1f;      // Probably choose atmostpheric pressure
    [SerializeField] Vector3 externalAccelerations = new(0, -9.8f, 0);      // Just gravity, for now at least

    [Header("Buffer & shader object properties")]
    // Main compute shader for calulating update to particles
    [SerializeField] ComputeShader SPHComputeShader;

    //  Make sure there's at least one thread group! TODO: just use NumParticles instead?
    int numThreadGroups = Mathf.Max(1, MaxNumParticles / ThreadGroupSize);

    // For testing
    [SerializeField] RenderTexture tex;

    struct ParticleData
    {
        public Vector3 position;
        public Vector3 velocity;
    }
    const int sizeofParticleData = sizeof(float) * 3 * 2;

    // Particle positions and current velocities are given to the CS kernels as an input.
    //  At the end, position and velocity are set by the ComputePosAndVel kernel by integration.
    [SerializeField] ComputeBuffer particleDataInputBuffer;
    // Buffers for interim calculations
    [SerializeField] ComputeBuffer DBuf;
    [SerializeField] ComputeBuffer PGBuf;
    [SerializeField] ComputeBuffer VLBuf;
    // Buffer for the calculation results
    [SerializeField] ComputeBuffer particleDataOutputBuffer;


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
        //if (tex == null)
        //{
        //    tex = new RenderTexture(numParticles, numParticles, 0, RenderTextureFormat.ARGBFloat);
        //    tex.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
        //    tex.volumeDepth = numParticles;
        //    tex.enableRandomWrite = true;
        //    tex.wrapMode = TextureWrapMode.Clamp;
        //    tex.filterMode = FilterMode.Point;
        //    tex.Create();
        //}

        int sizeofVec3 = sizeof(float) * 3;

        // note: SubUpdates prevents RenderDoc from viewing the buffer contents

        // Allocate the maximum amount of particles so that we don't need to keep creating new buffers when the number of particles changes
        particleDataInputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleData);
        particleDataOutputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleData);

        DBuf = new ComputeBuffer(MaxNumParticles, sizeof(float));
        PGBuf = new ComputeBuffer(MaxNumParticles, sizeofVec3);
        VLBuf = new ComputeBuffer(MaxNumParticles, sizeofVec3);

        // Initialize buffers, using unity's particle system emission shape to choose random initial positions in a box
        //var psShape = particleSystem.shape;
        particleSystem.Emit(numParticles);
        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[numParticles];
        particleSystem.GetParticles(particles);

        ParticleData[] dataArr = new ParticleData[MaxNumParticles];
        Vector3[] vec3Arr = new Vector3[MaxNumParticles];
        float[] floatArr = new float[MaxNumParticles];
        for (int i = 0; i < MaxNumParticles; i++)
        {
            if (i < numParticles)
            {
                Vector3 pos = particles[i].position;
                if (float.IsNaN(pos.x) || float.IsNaN(pos.y) || float.IsNaN(pos.z))
                {
                    throw new System.Exception("Found a NaN! index " + i);
                }
                dataArr[i].position = particles[i].position;
            }
            else
            {
                dataArr[i].position = Vector3.zero;
            }

            dataArr[i].velocity = Vector3.zero;
            vec3Arr[i] = Vector3.zero;
            floatArr[i] = 0f;
        }

        particleDataInputBuffer.SetData(dataArr);
        particleDataOutputBuffer.SetData(dataArr);       // Initialize output with the same data as input
        // Initialize buffer contents to all zeros
        DBuf.SetData(floatArr);
        PGBuf.SetData(vec3Arr);
        VLBuf.SetData(vec3Arr);
    }

    void ReleaseBuffers()
    {
        particleDataInputBuffer.Release();
        particleDataOutputBuffer.Release();
        DBuf.Release();
        PGBuf.Release();
        VLBuf.Release();
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

    // Compute density
    //  Inputs: position buffer, _NumParticles, _ParticleMass.
    //  Outputs: density buffer
    void DispatchDensity()
    {
        int id = SPHComputeShader.FindKernel("ComputeDensity");
        // input
        SPHComputeShader.SetBuffer(id, dataInputBufID, particleDataInputBuffer);
        SPHComputeShader.SetInt(numParticlesID, numParticles);
        SPHComputeShader.SetFloat(particleMassID, particleMass);

        // output
        SPHComputeShader.SetBuffer(id, "D", DBuf);

        SPHComputeShader.Dispatch(id, numThreadGroups, 1, 1);
    }

    // Compute pressure gradient and velocity laplacian. Doing both together because
    //  their calculations are independent of each other. Includes calculation for pressure
    //
    // PG:
    //  Inputs: particle position (for kernel grad.), pressure, density,
    //      _PressureStiffness, _ReferenceDensity, _NumParticles, _ParticleMass
    //  Outputs: PG buf
    //
    // VL:
    //  Inputs: particle position (for kernel lapl.), velocity, density, _Viscosity, _NumParticles, _ParticleMass
    //  Outputs: VL buf
    void DispatchPGandVL()
    {
        int id = SPHComputeShader.FindKernel("ComputePGandVL");
        // input - common
        SPHComputeShader.SetBuffer(id, dataInputBufID, particleDataInputBuffer);
        SPHComputeShader.SetBuffer(id, "D", DBuf);
        SPHComputeShader.SetInt(numParticlesID, numParticles);
        SPHComputeShader.SetFloat(particleMassID, particleMass);
        // input for PG
        SPHComputeShader.SetFloat(pressureStiffnessID, pressureStiffness);
        SPHComputeShader.SetFloat(referenceDensityID, referenceDensity);
        // input for VL
        SPHComputeShader.SetFloat(viscosityID, viscosity);

        // output
        SPHComputeShader.SetBuffer(id, "PG", PGBuf);
        SPHComputeShader.SetBuffer(id, "VL", VLBuf);

        SPHComputeShader.Dispatch(id, numThreadGroups, 1, 1);
    }

    // Compute particle acceleration and populate position and velocity buffers
    //  Inputs: position, velocity, PG, VL, _ExternalAccelerations, _ParticleMass, _DeltaTime
    //
    //  Outputs: new position and velocity
    void DispatchPosAndVel()
    {
        int id = SPHComputeShader.FindKernel("ComputePosAndVel");
        // input
        SPHComputeShader.SetBuffer(id, dataInputBufID, particleDataInputBuffer);
        SPHComputeShader.SetBuffer(id, "PG", PGBuf);
        SPHComputeShader.SetBuffer(id, "VL", VLBuf);
        SPHComputeShader.SetVector(extAccelerationsID, externalAccelerations);
        SPHComputeShader.SetFloat(particleMassID, particleMass);
        //TODO: timestep!! Should be constant for CFL??? (or have cfl adjust per frame?)
        //  But for physics simulation, should be constant! Or else it's dependent on framerate
        SPHComputeShader.SetFloat(deltaTimeID, 0.001f);

        // output
        SPHComputeShader.SetBuffer(id, dataOutputBufID, particleDataOutputBuffer);

        SPHComputeShader.Dispatch(id, numThreadGroups, 1, 1);
    }

    // Dispatch a job to the GPU to run a new SPH simulation step
    void SimulateParticles()
    {
        if (numThreadGroups > 65535)
        {
            Debug.LogWarning("Number of thread groups (" + numThreadGroups + ") would exceed the maximum of 65535.");
            numThreadGroups = 65535;
        }

        // Order of calculation pipeline: density, pressure, pressure gradient and velocity laplacian (independent), 
        //  then finally acceleration, position and velocity.
        DispatchDensity();
        DispatchPGandVL();
        DispatchPosAndVel();

        //// Position & velocity buffer is now updated, so set particles from that.
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
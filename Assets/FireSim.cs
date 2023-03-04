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
        particlePositionIOBufID = Shader.PropertyToID("_ParticlePositionIOBuf"),
        particleVelocityInputBufID = Shader.PropertyToID("_ParticleVelocityInputBuf"), 
        particleAccelIOBufID = Shader.PropertyToID("_ParticleAccelIOBuf"),

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
    //[SerializeField] Texture3D tex;

    // Particle positions and current velocities are given to the acceleration compute shader as an input.
    //  Position is then set by the ComputePosition kernel by integration.
    [SerializeField] ComputeBuffer particlePositionIOBuf;
    [SerializeField] ComputeBuffer particleVelocityInputBuf;
    // Buffer for the calculation results
    [SerializeField] ComputeBuffer particleAccelIOBuf;

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

        // Define the particle position buffer as structured, with dynamic, unsynchronized access.
        // Allocate the maximum amount of particles so that we don't need to keep creating new buffers when the number of particles changes
        particlePositionIOBuf = new ComputeBuffer(MaxNumParticles, sizeofVec3, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
        particleVelocityInputBuf = new ComputeBuffer(MaxNumParticles, sizeofVec3, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
        // note: SubUpdates prevents RenderDoc from viewing the buffer contents
        particleAccelIOBuf = new ComputeBuffer(MaxNumParticles, sizeofVec3, ComputeBufferType.Default);

        // Initialize buffers, using unity's particle system emission shape to choose random initial positions in a box
        //var psShape = particleSystem.shape;
        particleSystem.Emit(numParticles);
        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[numParticles];
        particleSystem.GetParticles(particles);

        Vector3[] posArr = new Vector3[MaxNumParticles];
        Vector3[] velArr = new Vector3[MaxNumParticles];
        Vector3[] accArr = new Vector3[MaxNumParticles];
        for (int i = 0; i < MaxNumParticles; i++)
        {
            if (i < numParticles)
                posArr[i] = particles[i].position;
            else
                posArr[i] = Vector3.zero;
            velArr[i] = Vector3.zero;
            accArr[i] = Vector3.zero;
        }
        particlePositionIOBuf.SetData(posArr);
        particleVelocityInputBuf.SetData(velArr);
        particleAccelIOBuf.SetData(accArr);
    }

    void ReleaseBuffers()
    {
        particlePositionIOBuf.Release();
        particleVelocityInputBuf.Release();
        particleAccelIOBuf.Release();
    }

    // Write current particle positions and velocities for acceleration CS
    void WriteParticleDataToGPU()
    {
        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[numParticles];
        particleSystem.GetParticles(particles);



        //TODO: make compute buffer mode Dynamic for all buffers and just use SetData instead of BeginWrite so we can see buffer contents in RenderDoc!!!!!!



        NativeArray<Vector3> GPUPosArray = particlePositionIOBuf.BeginWrite<Vector3>(0, numParticles);
        NativeArray<Vector3> GPUVelArray = particleVelocityInputBuf.BeginWrite<Vector3>(0, numParticles);
        for (int i = 0; i < numParticles; i++)
        {
            GPUPosArray[i] = particles[i].position;
            GPUVelArray[i] = particles[i].velocity;
        }
        particleVelocityInputBuf.EndWrite<Vector3>(numParticles);
        particlePositionIOBuf.EndWrite<Vector3>(numParticles);
    }

    // Dispatch a job to the GPU to run a new SPH simulation step
    void SimulateParticles()
    {
        int kernelID = SPHComputeShader.FindKernel("ComputeAcceleration");

        // Set debug property because it blows up if you don't
        SPHComputeShader.SetTexture(kernelID, Shader.PropertyToID("Result"), tex);
        SPHComputeShader.SetBuffer(kernelID, particlePositionIOBufID, particlePositionIOBuf);
        SPHComputeShader.SetBuffer(kernelID, particleVelocityInputBufID, particleVelocityInputBuf);
        SPHComputeShader.SetBuffer(kernelID, particleAccelIOBufID, particleAccelIOBuf);

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
        SPHComputeShader.SetBuffer(kernelID, particlePositionIOBufID, particlePositionIOBuf);
        SPHComputeShader.SetBuffer(kernelID, particleVelocityInputBufID, particleVelocityInputBuf);
        SPHComputeShader.SetBuffer(kernelID, particleAccelIOBufID, particleAccelIOBuf);

        SPHComputeShader.SetFloat(deltaTimeID, Time.deltaTime);
        SPHComputeShader.Dispatch(kernelID, numThreadGroups, 1, 1);

        // Positions buffer is now updated, set particle positions.
        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[numParticles];
        particleSystem.GetParticles(particles);

        Vector3[] posArray = new Vector3[numParticles];
        particlePositionIOBuf.GetData(posArray);

        for (int i = 0; i < numParticles; i++)
        {
            particles[i].position = posArray[i];
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
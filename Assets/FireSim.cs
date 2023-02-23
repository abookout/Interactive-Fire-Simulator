using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;

public class Main : MonoBehaviour
{
    // This should match with the SPH CS!
    const int ThreadGroupSize = 64;

    // Buffer and parameter IDs
    static readonly int
        particlePositionInputBufID = Shader.PropertyToID("_ParticlePositionInputBuf"),
        particleAccelOutputBufID = Shader.PropertyToID("_ParticleAccelOutputBuf"),

        maxNumParticlesID = Shader.PropertyToID("_MaxNumParticles"),
        particleMassID = Shader.PropertyToID("_ParticleMass"),
        pressureStiffnessID = Shader.PropertyToID("_PressureStiffness"),
        referenceDensityID = Shader.PropertyToID("_ReferenceDensity"),
        gravityID = Shader.PropertyToID("_Gravity");

    [Header("Set in inspector")]
    [SerializeField] new ParticleSystem particleSystem;
    [SerializeField] int maxNumParticles = 32;
    [SerializeField] float particleMass = 0.1f;
    [SerializeField] float pressureStiffness = 1f;
    [SerializeField] float referenceDensity = 1f;      // Probably choose atmostpheric pressure
    [SerializeField] float gravityStrength = 9.8f;

    // Main compute shader for calulating update to particle accelerations
    [Header("Buffer & shader object properties")]
    [SerializeField] ComputeShader SPHComputeShader;

    // For testing
    [SerializeField] RenderTexture tex;
    //[SerializeField] Texture3D tex;

    // Particle positions are given to the compute shader as an input
    [SerializeField] ComputeBuffer particlePositionInputBuf;
    // Buffer for the calculation results
    [SerializeField] ComputeBuffer particleAccelOutputBuf;

    // Using OnEnable instead of start or awake so that the buffers are refreshed every hot reload
    private void OnEnable()
    {
        InitializeBuffers();
        SetupParticleSystem();
    }
    private void OnDisable()
    {
        ReleaseBuffers();
    }
    private void FixedUpdate()
    {
        UpdateParticleAccelerations();
    }

    // Render results of compute shader to render tex for testing
    //private void OnRenderImage(RenderTexture source, RenderTexture destination)
    //{
    //    int texWidth = 256;
    //    if (tex == null)
    //    {
    //        tex = new RenderTexture(texWidth, texWidth, 24);
    //        tex.enableRandomWrite = true;
    //        tex.Create();
    //    }

    //    Graphics.Blit(tex, destination);
    //}


    void InitializeBuffers()
    {
        if (tex == null)
        {
            tex = new RenderTexture(maxNumParticles, maxNumParticles, 0, RenderTextureFormat.ARGBFloat);
            tex.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            tex.volumeDepth = maxNumParticles;
            tex.enableRandomWrite = true;
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Point;
            tex.Create();

            //tex = new Texture3D(256, 256, 256, TextureFormat.RGBA32, mipChain: false, createUninitialized: true);
            //tex.wrapMode = TextureWrapMode.Clamp;
            //tex.filterMode = FilterMode.Point;
            //tex.Apply();
        }

        int mainID = SPHComputeShader.FindKernel("CSComputeAccel");

        // TODO: is there a better way of computing this?
        int sizeofVec3 = sizeof(float) * 3;

        // Define the particle position buffer as structured, with dynamic, unsynchronized access
        particlePositionInputBuf = new ComputeBuffer(maxNumParticles, sizeofVec3, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
        particleAccelOutputBuf = new ComputeBuffer(maxNumParticles, sizeofVec3, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);


        //TODO: write constants buffer and maybe also positions buffer here?
    }
    
    // Dispatch a job to the GPU to run a new SPH simulation step
    void UpdateParticleAccelerations()
    {
        int mainID = SPHComputeShader.FindKernel("CSComputeAccel");

        // Set debug property because it blows up if you don't
        SPHComputeShader.SetTexture(mainID, Shader.PropertyToID("Result"), tex);

        SPHComputeShader.SetBuffer(mainID, particlePositionInputBufID, particlePositionInputBuf);
        SPHComputeShader.SetBuffer(mainID, particleAccelOutputBufID, particleAccelOutputBuf);

        // Set constants
        SPHComputeShader.SetInt(maxNumParticlesID, maxNumParticles);
        SPHComputeShader.SetFloat(particleMassID, particleMass);
        SPHComputeShader.SetFloat(pressureStiffnessID, pressureStiffness);
        SPHComputeShader.SetFloat(referenceDensityID, referenceDensity);
        SPHComputeShader.SetFloat(gravityID, gravityStrength);
        //  Make sure there's at least one thread group!


        // This works with [numthreads(8, 8, 1)]
        //SPHComputeShader.Dispatch(0, tex.width / 8, tex.height / 8, 1);

        // This works with [numthreads(64, 1, 1)]
        //SPHComputeShader.Dispatch(0, tex.width*4, 1, 1);

        // To process x items, with a kernel group size of ThreadGroupSize*1*1, call Dispatch(x/ThreadGroupSize, 1, 1)!!
        // this is good!
        //SPHComputeShader.Dispatch(0, 256*256 / ThreadGroupSize, 1, 1);

        //SPHComputeShader.Dispatch(0, maxNumParticles * maxNumParticles / ThreadGroupSize, 1, 1);

        // This works with [numthreads(16, 1, 1)]
        //SPHComputeShader.Dispatch(0, tex.width * 16, 1, 1);

        // Squared num particles because of the size of the test texture
        int numThreadGroups = Mathf.Max(1, maxNumParticles * maxNumParticles * maxNumParticles / ThreadGroupSize);

        SPHComputeShader.Dispatch(mainID, numThreadGroups, 1, 1);
    }

    void ReleaseBuffers()
    {
        particlePositionInputBuf.Release();
        particleAccelOutputBuf.Release();
    }

    // Needs the buffers to have been initialized first
    void SetupParticleSystem()
    {
        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[maxNumParticles];
        //particles = new ParticleSystem.Particle[maxNumParticles];
        // This is how you set particle system struct b/c it's a struct that exposes the underlying implementation
        var main = particleSystem.main;
        main.maxParticles = maxNumParticles;

        //var shape = particleSystem.shape;
        particleSystem.Emit(maxNumParticles);

        // Populate particles array from system
        particleSystem.GetParticles(particles);

        NativeArray<Vector3> GPUArray = particlePositionInputBuf.BeginWrite<Vector3>(0, maxNumParticles);
        for (int i = 0; i < maxNumParticles; i++)
        {
            GPUArray[i] = particles[i].position;
        }
        particlePositionInputBuf.EndWrite<Vector3>(maxNumParticles);

        // TODO: Set position and scale of a bounding box for the fire to exist for the sake of keeping the fire localized (??)

        //particleSystem.SetParticles(particles, maxNumParticles);
    }
}

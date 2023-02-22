using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Main : MonoBehaviour
{
    // This should match with the SPH CS!
    const int ThreadGroupSize = 64;

    [Header("Parameters to set in inspector")]
    [SerializeField] int maxNumParticles = 100;
    [SerializeField] float particleMass = 0.1f;
    [SerializeField] float gravityStrength = 9.8f;

    [Header("Buffer & shader object properties")]
    // Buffer and parameter IDs
    static readonly int
        particlePositionInputBufID = Shader.PropertyToID("_ParticlePositionInputBuf"),
        particleAccelOutputBufID = Shader.PropertyToID("_ParticleAccelOutputBuf"),

        maxNumParticlesID = Shader.PropertyToID("_MaxNumParticles"),
        particleMassID = Shader.PropertyToID("_ParticleMass"),
        gravityID = Shader.PropertyToID("_Gravity");

    // Main compute shader for calulating update to particle accelerations
    [SerializeField] ComputeShader SPHComputeShader;

    // For testing
    [SerializeField] RenderTexture tex;

    // Particle positions are given to the compute shader as an input
    [SerializeField] ComputeBuffer particlePositionInputBuf;
    // Buffer for the calculation results
    [SerializeField] ComputeBuffer particleAccelOutputBuf;

    // Using OnEnable instead of start or awake so that the buffers are refreshed every hot reload
    private void OnEnable()
    {
        InitializeBuffers();
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
    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        int texWidth = 256;
        if (tex == null)
        {
            tex = new RenderTexture(texWidth, texWidth, 24);
            tex.enableRandomWrite = true;
            tex.Create();
        }

        Graphics.Blit(tex, destination);
    }


    void InitializeBuffers()
    {
        if (tex == null)
        {
            tex = new RenderTexture(256, 256, 24);
            tex.enableRandomWrite = true;
            tex.Create();
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
        int numThreadGroups = Mathf.Max(1, maxNumParticles * maxNumParticles / ThreadGroupSize);

        SPHComputeShader.Dispatch(mainID, numThreadGroups, 1, 1);
    }

    void ReleaseBuffers()
    {
        particlePositionInputBuf.Release();
        particleAccelOutputBuf.Release();
    }

    void SetupParticleSystem()
    {

    }
}

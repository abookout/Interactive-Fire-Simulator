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
        constantsBufId = Shader.PropertyToID("_ParticleConstantsBuf"),
        particlePositionInputBufID = Shader.PropertyToID("_ParticlePositionInputBuf"),
        //fieldDataId = Shader.PropertyToID("_FieldDataBuf"),
        particleAccelOutputBufID = Shader.PropertyToID("_ParticleAccelOutputBuf");

    // Main compute shader for calulating update to particle accelerations
    [SerializeField] ComputeShader SPHComputeShader;

    // For testing
    [SerializeField] RenderTexture tex;
    [SerializeField]
    struct SPHConstants
    {
        int maxNumParticles;
        float particleMass;
        float gravity;
    }
    const int sizeOfSPHConstants = sizeof(int) + sizeof(float) * 2;     // Remember to update when changing properties!!!!

    // Buffer for constants to pass to shader
    [SerializeField] ComputeBuffer constantsBuf;
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

    // Render results of compute shader to render tex for testing
    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (tex == null)
        {
            tex = new RenderTexture(256, 256, 24);
            tex.enableRandomWrite = true;
            tex.Create();
        }

        //int main = computeShader.FindKernel("CSMain");
        SPHComputeShader.SetTexture(0, "Result", tex);
        SPHComputeShader.Dispatch(0, tex.width / 8, tex.height / 8, 1);

        Graphics.Blit(tex, destination);
    }

    private void FixedUpdate()
    {
        UpdateParticleAccelerations();
    }

    void InitializeBuffers()
    {
        int mainID = SPHComputeShader.FindKernel("CSComputeAccel");

        // TODO: is there a better way of computing this?
        int sizeofVec3 = sizeof(float) * 3;

        constantsBuf = new ComputeBuffer(maxNumParticles, sizeOfSPHConstants, ComputeBufferType.Default, ComputeBufferMode.Immutable);

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
        SPHComputeShader.SetBuffer(mainID, Shader.PropertyToID("Results"), )

        // Not using SetConstantBuffer for the constants struct because we want to be able to pass a struct (so need StructuredBuffer, aka ComputeBufferType.Default)
        SPHComputeShader.SetBuffer(mainID, constantsBufId, constantsBuf);
        SPHComputeShader.SetBuffer(mainID, particlePositionInputBufID, particlePositionInputBuf);
        SPHComputeShader.SetBuffer(mainID, particleAccelOutputBufID, particleAccelOutputBuf);

        //TODO: does this make sense?
        // The number of thread groups is determined by spreading out the max number of particles over the number of threads per group.
        //  Make sure there's at least one thread group!
        //int numThreadGroups = Mathf.Max(1, maxNumParticles / ThreadGroupSize);

        //SPHComputeShader.Dispatch(mainID, numThreadGroups, 1, 1);
        SPHComputeShader.Dispatch(mainID, 1, 1, 1);

    }

    void ReleaseBuffers()
    {
        constantsBuf.Release();
        particlePositionInputBuf.Release();
        particleAccelOutputBuf.Release();
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Main : MonoBehaviour
{
    [Header("Parameters to set in inspector")]
    [SerializeField] int maxNumParticles;
    [SerializeField] float particleMass;
    [SerializeField] float gravityStrength;

    [Header("Buffer & shader object properties")]

    static readonly int 
        particlePositionsId = Shader.PropertyToID("_ParticlePositionInputBuf"),
        fieldDataId = Shader.PropertyToID("_FieldDataBuf"),
        particleConstantsId = Shader.PropertyToID("_ParticleConstantsBuf");

    public ComputeShader SPHComputeShader;

    // For testing
    public RenderTexture tex;

    // Particle positions are given to the compute shader as an input
    public ComputeBuffer particlePositionInputBuf;

    // The field data is caluclated by the compute shader for 
    //public ComputeBuffer fieldDataBuf;

    //TODO: probably going to remove this and just track them in shader with groupshared arrays
    struct FieldData
    {
        Vector3 velocity;
        Vector3 density;
        Vector3 pressure;
    }
    const int sizeOfFieldData = sizeof(float) * 3 * 3;     // Remember to update if changing FieldData's properties!!!!


    // Using OnEnable instead of start or awake so that the buffers are refreshed every hot reload
    private void OnEnable()
    {
        InitializeBuffers();
    }
    private void OnDisable()
    {
        ReleaseBuffers();
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // Render results of compute shader to render tex for testing
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


    void InitializeBuffers()
    {
        int mainID = SPHComputeShader.FindKernel("CSComputeAccel");

        // TODO: is there a better way of computing this?
        int sizeofVec3 = sizeof(float) * 3;

        // Define the particle position buffer as structured, with dynamic, unsynchronized access
        particlePositionInputBuf = new ComputeBuffer(maxNumParticles, sizeofVec3, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
        SPHComputeShader.SetBuffer(mainID, particlePositionsId, particlePositionInputBuf);

        //fieldDataBuf = new ComputeBuffer(maxNumParticles, sizeOfFieldData, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
    }

    void ReleaseBuffers()
    {
        particlePositionInputBuf.Release();
        //fieldDataBuf.Release();
    }
}

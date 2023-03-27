using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;

// For setting up test scenarios
[System.Serializable]
public struct DebugParticleData
{
    public bool fixInSpace;
    public Vector3 position;
    public Vector3 velocity;
    public float temperature;
}
[System.Serializable]
public struct DebugConfiguration
{
    public string description;
    public List<DebugParticleData> particles;
}

public class FireSim : MonoBehaviour
{
    // This should match with the SPH CS!
    const int ThreadGroupSize = 64;
    const int MaxNumParticles = 4096;

    // Buffer and parameter IDs
    static readonly int
        // Buffers
        dataInputBufID = Shader.PropertyToID("_DataInputBuffer"),
        densityBufID = Shader.PropertyToID("_DensityBuf"),
        pressureGradientBufID = Shader.PropertyToID("_PressureGradientBuf"),
        diffusionBufID = Shader.PropertyToID("_DiffusionBuf"),
        dataOutputBufID = Shader.PropertyToID("_DataOutputBuffer"),

        // Parameters
        deltaTimeID = Shader.PropertyToID("_DeltaTime"),
        numParticlesID = Shader.PropertyToID("_NumParticles"),
        particleMassID = Shader.PropertyToID("_ParticleMass"),
        pressureStiffnessID = Shader.PropertyToID("_PressureStiffness"),
        viscosityID = Shader.PropertyToID("_Viscosity"),
        referenceDensityID = Shader.PropertyToID("_ReferenceDensity"),
        heatDiffusionID = Shader.PropertyToID("_HeatDiffusionRate"),
        extAccelerationsID = Shader.PropertyToID("_ExternalAccelerations");

    // Array is saved over updates so it doesn't need keep being allocated
    ParticleSystem.Particle[] particlesArr;

    // It's a Vector4 because of how the Unity particle system tracks custom particle data. Only the first component is used. 
    List<Vector4> particleTemperatureArr = new List<Vector4>();

    [Header("Debug configurations")]
    public List<DebugConfiguration> debugConfigurations;
    public int selectedDebugConfiguration = 0;          // 0 means none are selected
    public DebugConfiguration? currentDebugConfiguration => selectedDebugConfiguration == 0 ? null : debugConfigurations[selectedDebugConfiguration - 1];

    [Header("Particle system info")]
    [SerializeField] new ParticleSystem particleSystem;
    [SerializeField, Range(1, MaxNumParticles)] int numParticles = 256;

    [Header("SPH simulation DOFs")]
    [SerializeField, Min(0)] float particleMass = 0.1f;         // Make it larger to slow down the simulation
    [SerializeField, Min(0)] float viscosity = 15.8f;           // Kinematic viscosity of air is 15.8 m^2/s (text p.276)
    [SerializeField, Min(0)] float pressureStiffness = 1f;
    [SerializeField, Min(0)] float referenceDensity = 1.18f;                        // Density of air is 1.18 kg/m^3
    [SerializeField, Min(0)] float ambientTemperature = 25f;  // degrees Celcius. TODO: is this only useful for initializing particle temp?
    [SerializeField, Min(0)] float temperatureDiffusionRate = 1f;
    [SerializeField] Vector3 externalAccelerations = new(0, 0, 0);

    [Header("Simulation spawn volume and bounds")]
    [SerializeField] bool spawnEvenlySpaced = true;
    [SerializeField] bool usePeriodicBoundary = true;
    [SerializeField] float spawnVolumeWidth = 2f;
    [SerializeField] Bounds particleBounds;

    [Header("Buffer & shader object properties")]
    // Main compute shader for calulating update to particles
    [SerializeField] ComputeShader SPHComputeShader;


    //  Make sure there's at least one thread group!
    int numThreadGroups;

    struct ParticleData
    {
        public Vector3 position;
        public Vector3 velocity;
        public float temperature;
    }
    const int sizeofParticleData = sizeof(float) * 7;

    // Particle positions and current velocities are given to the CS kernels as an input.
    //  At the end, position and velocity are set by the ComputePosAndVel kernel by integration.
    [SerializeField] ComputeBuffer particleDataInputBuffer;
    // Buffers for interim calculations
    [SerializeField] ComputeBuffer densityBuf;
    [SerializeField] ComputeBuffer pressureGradientBuf;
    [SerializeField] ComputeBuffer diffusionBuf;
    // Buffer for the calculation results
    [SerializeField] ComputeBuffer particleDataOutputBuffer;

    public void RespawnParticles()
    {
        if (selectedDebugConfiguration != 0)
        {
            DebugConfiguration config = currentDebugConfiguration.Value;
            // If using a debug configuration don't respawn as normal, just populate from configuration
            particleSystem.GetParticles(particlesArr);

            for (int i = 0; i < config.particles.Count; i++)
            {
                DebugParticleData p = config.particles[i];
                particlesArr[i].position = p.position;
                particlesArr[i].velocity = p.velocity;
            }

            particleSystem.SetParticles(particlesArr, config.particles.Count);
            return;
        }

        // Clear and repopulate particles
        particleSystem.Clear();
        UpdateParticleCount();

        if (spawnEvenlySpaced)
        {
            EvenlySpaceParticles();
        }
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireCube(particleSystem.transform.position, new Vector3(spawnVolumeWidth, spawnVolumeWidth, spawnVolumeWidth));

        Gizmos.color = Color.blue;
        Gizmos.DrawWireCube(particleSystem.transform.position, particleBounds.extents * 2);
    }


    // Using OnEnable instead of start or awake so that the buffers are refreshed every hot reload
    private void OnEnable()
    {
        //Unity.Collections.LowLevel.Unsafe.UnsafeUtility.SetLeakDetectionMode(NativeLeakDetectionMode.EnabledWithStackTrace);

        if (selectedDebugConfiguration == 0)
        {
            // Initialize as normal
            Initialize();
        }
        else
        {
            // Using a debug configuration
            InitializeFromDebugConfig();
        }
    }
    private void OnDisable()
    {
        ReleaseBuffers();
    }


    readonly float fpsDrawInterval = 0.5f;
    int fpsLastVal = -1;
    float fpsLastDrawTime = 0;
    private void OnGUI()
    {
        GUIStyle style = new GUIStyle();
        style.fontSize = 20;
        style.normal.textColor = Color.white;
        GUI.TextField(new Rect(5, 5, 20, 100), "FPS " + fpsLastVal, style);
    }

    void UpdateParticleCount()
    {
        particlesArr = new ParticleSystem.Particle[numParticles];
        // It has changed, so need to either destroy some or instantiate more
        if (particleSystem.particleCount > numParticles)
        {
            // Only set numParticles back into the array (which is less than before)
            particleSystem.GetParticles(particlesArr);
            particleSystem.SetParticles(particlesArr, numParticles);
        }
        else
        {
            // particleCount < numParticles, so emit the difference.
            // Update volume size before spawning new particles in case it changed
            var psShape = particleSystem.shape;
            psShape.scale = new Vector3(spawnVolumeWidth, spawnVolumeWidth, spawnVolumeWidth);

            //TODO: it makes a big difference where these are spawned in terms of the pressure gradient!
            particleSystem.Emit(numParticles - particleSystem.particleCount);
        }
    }

    bool isFirstUpdate = true;
    private void Update()
    {
        if (Time.time > fpsLastDrawTime + fpsDrawInterval)
        {
            fpsLastVal = (int)(1 / Time.smoothDeltaTime);
            fpsLastDrawTime = Time.time;
        }

        // Make sure the number of particles hasn't changed.
        // If using a debug configuration, don't update particle count as normal
        if (selectedDebugConfiguration == 0 && particleSystem.particleCount != numParticles)
        {
            UpdateParticleCount();
        }

        if (isFirstUpdate)
        {
            // Skip update on first frame just for making gpu debugging nicer (it's hard to capture the first frame in RenderDoc)
            isFirstUpdate = false;
            return;
        }

        // Send positions and velocities to GPU
        WriteParticleDataToGPU();

        // Perform SPH calculations
        SimulateParticles();

        // TODO: This stuff is just for testing
        ParticleSystem.Particle[] ps = new ParticleSystem.Particle[1];
        float[] d = new float[1];
        Vector3[] pg = new Vector3[1];
        Vector3[] vl = new Vector3[1];
        particleSystem.GetParticles(ps);
        densityBuf.GetData(d);
        pressureGradientBuf.GetData(pg);
        diffusionBuf.GetData(vl);

        Debug.Log("Particle 0:\n\tPosition: " + ps[0].position + "\tVelocity: " + ps[0].velocity + "\tDensity: " + d[0] + "\n\tPressure Gradient: " + pg[0] + "\t\tDiffusion: " + vl[0]);
    }

    // Initialize particle system and buffers for GPU
    void Initialize()
    {
        int sizeofVec3 = sizeof(float) * 3;

        // note: SubUpdates prevents RenderDoc from viewing the buffer contents

        // Allocate the maximum amount of particles so that we don't need to keep creating new buffers when the number of particles changes
        particleDataInputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleData);
        particleDataOutputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleData);

        densityBuf = new ComputeBuffer(MaxNumParticles, sizeof(float));
        pressureGradientBuf = new ComputeBuffer(MaxNumParticles, sizeofVec3);
        diffusionBuf = new ComputeBuffer(MaxNumParticles, sizeofVec3);

        // Initialize buffers, using unity's particle system emission shape to choose random initial positions in a box
        //var psShape = particleSystem.shape;
        var psMain = particleSystem.main;
        psMain.maxParticles = MaxNumParticles;

        var psShape = particleSystem.shape;
        psShape.scale = new Vector3(spawnVolumeWidth, spawnVolumeWidth, spawnVolumeWidth);

        // Enable custom data for tracking temperature
        var psCustomData = particleSystem.customData;
        psCustomData.enabled = true;
        psCustomData.SetMode(ParticleSystemCustomData.Custom1, ParticleSystemCustomDataMode.Vector);
        psCustomData.SetVectorComponentCount(ParticleSystemCustomData.Custom1, 1);

        // Spawn in the initial particles
        particlesArr = new ParticleSystem.Particle[numParticles];

        particleSystem.Emit(numParticles);
        particleSystem.GetParticles(particlesArr);
        particleSystem.GetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);

        // Evenly space them
        if (spawnEvenlySpaced)
        {
            EvenlySpaceParticles();
        }

        ParticleData[] particleDataArr = new ParticleData[MaxNumParticles];
        Vector3[] vec3Arr = new Vector3[MaxNumParticles];
        float[] floatArr = new float[MaxNumParticles];
        for (int i = 0; i < MaxNumParticles; i++)
        {
            if (i < numParticles)
            {
                particleDataArr[i].position = particlesArr[i].position;
                particleDataArr[i].temperature = ambientTemperature;
                particleTemperatureArr[i] = new Vector4(ambientTemperature, 0, 0, 0);
            }
            else
            {
                particleDataArr[i].position = Vector3.zero;
                particleTemperatureArr[i] = Vector4.zero;
            }

            particleDataArr[i].velocity = Vector3.zero;
            vec3Arr[i] = Vector3.zero;
            floatArr[i] = 0f;
        }

        particleSystem.SetParticles(particlesArr);
        particleSystem.SetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);

        particleDataInputBuffer.SetData(particleDataArr);
        particleDataOutputBuffer.SetData(particleDataArr);       // Initialize output with the same data as input
        // Initialize buffer contents to all zeros
        densityBuf.SetData(floatArr);
        pressureGradientBuf.SetData(vec3Arr);
        diffusionBuf.SetData(vec3Arr);
    }

    // Initialize as normal except populate particle data from the selected debug configuration
    void InitializeFromDebugConfig()
    {
        int sizeofVec3 = sizeof(float) * 3;

        // Allocate the maximum amount of particles so that we don't need to keep creating new buffers when the number of particles changes
        particleDataInputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleData);
        particleDataOutputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleData);

        densityBuf = new ComputeBuffer(MaxNumParticles, sizeof(float));
        pressureGradientBuf = new ComputeBuffer(MaxNumParticles, sizeofVec3);
        diffusionBuf = new ComputeBuffer(MaxNumParticles, sizeofVec3);

        // Initialize buffers, using unity's particle system emission shape to choose random initial positions in a box
        //var psShape = particleSystem.shape;
        var psMain = particleSystem.main;
        psMain.maxParticles = MaxNumParticles;

        var psShape = particleSystem.shape;
        psShape.scale = new Vector3(spawnVolumeWidth, spawnVolumeWidth, spawnVolumeWidth);

        DebugConfiguration curDebugConfig = (DebugConfiguration)currentDebugConfiguration;
        numParticles = curDebugConfig.particles.Count;

        // Spawn in the particles from the debug config
        particlesArr = new ParticleSystem.Particle[numParticles];
        particleSystem.Emit(numParticles);
        particleSystem.GetParticles(particlesArr);
        //particleSystem.GetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);

        // Enable custom data for tracking temperature
        var psCustomData = particleSystem.customData;
        psCustomData.enabled = true;
        psCustomData.SetMode(ParticleSystemCustomData.Custom1, ParticleSystemCustomDataMode.Vector);
        psCustomData.SetVectorComponentCount(ParticleSystemCustomData.Custom1, 1);

        ParticleData[] particleDataArr = new ParticleData[MaxNumParticles];
        Vector3[] vec3Arr = new Vector3[MaxNumParticles];
        float[] floatArr = new float[MaxNumParticles];
        for (int i = 0; i < MaxNumParticles; i++)
        {
            if (i < numParticles)
            {
                DebugParticleData particle = curDebugConfig.particles[i];
                particleDataArr[i].position = particle.position;
                particleDataArr[i].velocity = particle.velocity;
                particleDataArr[i].temperature = particle.temperature;
                // Populate data in ParticleSystem so update uses that
                particlesArr[i].position = particle.position;
                particlesArr[i].velocity = particle.velocity;
                particleTemperatureArr[i] = new Vector4(particle.temperature, 0, 0, 0);
            }
            else
            {
                particleDataArr[i].position = Vector3.zero;
                particleDataArr[i].velocity = Vector3.zero;
            }

            // Fill with zeros so the buffers aren't initialized with random data
            vec3Arr[i] = Vector3.zero;
            floatArr[i] = 0f;
        }

        particleSystem.SetParticles(particlesArr);
        particleSystem.SetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);

        particleDataInputBuffer.SetData(particleDataArr);
        particleDataOutputBuffer.SetData(particleDataArr);       // Initialize output with the same data as input
        // Initialize buffer contents to all zeros
        densityBuf.SetData(floatArr);
        pressureGradientBuf.SetData(vec3Arr);
        diffusionBuf.SetData(vec3Arr);
    }

    void ReleaseBuffers()
    {
        particleDataInputBuffer.Release();
        particleDataOutputBuffer.Release();
        densityBuf.Release();
        pressureGradientBuf.Release();
        diffusionBuf.Release();
    }

    // Write current particle positions and velocities for acceleration CS
    void WriteParticleDataToGPU()
    {
        particleSystem.GetParticles(particlesArr);
        particleSystem.GetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);

        //TODO: maintain position and velocity arrays so don't need to access from particle system? Not sure if this is a problem
        ParticleData[] dataArray = new ParticleData[numParticles];

        for (int i = 0; i < numParticles; i++)
        {
            dataArray[i].position = particlesArr[i].position;
            dataArray[i].velocity = particlesArr[i].velocity;
            dataArray[i].temperature = particleTemperatureArr[i].x;
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

    // For initial spawn, create particles evenly spaced in the spawn volume
    void EvenlySpaceParticles()
    {
        // Make sure the array is populated with the correct particle data
        particlesArr = new ParticleSystem.Particle[numParticles];
        particleSystem.GetParticles(particlesArr);

        // If numParticles doesn't have an exact cube root, find its next largest perfect cube and use that for spacing instead
        //  so that we can fit all the particles on the lattice (even though there will be some empty space)
        float particleSpacingFloat = Mathf.Pow(numParticles, 1.0f / 3.0f);
        // Using ceilToInt will make sure that the spacing has a perfect cube root
        int particleSpacing = Mathf.CeilToInt(particleSpacingFloat);

        int i = 0;
        for (int y = 0; y < particleSpacing; y++)
        {
            for (int x = 0; x < particleSpacing; x++)
            {
                for (int z = 0; z < particleSpacing; z++)
                {
                    // Distribute particles evenly, where the first and last are on the edge boundaries of the spawn volume
                    Vector3 pos = new Vector3(x, y, z);
                    ParticleSystem.Particle p = particlesArr[i];

                    // Change the position's range from 0..(particleSpacing-1) to 0..spawnVolumeWidth but offset to center the volume
                    p.position = MapRange(pos, 0, particleSpacing - 1, -spawnVolumeWidth / 2, spawnVolumeWidth / 2);
                    particlesArr[i] = p;
                    i++;

                    // Hacky way of breaking out of the nested loop - this is the case when numParticles is not a perfect cube root
                    if (i >= numParticles) break;
                }
                if (i >= numParticles) break;
            }
            if (i >= numParticles) break;
        }

        particleSystem.SetParticles(particlesArr, numParticles);
    }

    float MapRange(float val, float from1, float to1, float from2, float to2)
    {
        return (val - from1) / (to1 - from1) * (to2 - from2) + from2;
    }

    // MapRange of each of val's x, y, and z
    Vector3 MapRange(Vector3 val, float from1, float to1, float from2, float to2)
    {
        return new Vector3(
            MapRange(val.x, from1, to1, from2, to2),
            MapRange(val.y, from1, to1, from2, to2),
            MapRange(val.z, from1, to1, from2, to2)
            );
    }

    int RoundToNearest(float num, int nearest)
    {
        return Mathf.CeilToInt(num / nearest) * nearest;
    }

    // Dispatch a job to the GPU to run a new SPH simulation step
    void SimulateParticles()
    {
        // Pick a number of thread groups that divides numParticles into groups of size ThreadGroupSize
        numThreadGroups = Mathf.Max(1, Mathf.CeilToInt(numParticles / (float)ThreadGroupSize));

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

        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[numParticles];
        particleSystem.GetParticles(particles);

        //// Position & velocity buffer is now updated, so set particles from that.
        ParticleData[] dataArray = new ParticleData[numParticles];
        particleDataOutputBuffer.GetData(dataArray);

        for (int i = 0; i < numParticles; i++)
        {
            Vector3 newPosition = dataArray[i].position;
            Vector3 newVelocity = dataArray[i].velocity;

            ////// TODO: Testing: if particle got NaN'd, fix it so it's easier to find the problem
            if (float.IsNaN(newPosition.magnitude))
            {
                Debug.Log("Particle " + i + " had NaN position " + newPosition);
                newPosition = Vector3.zero;
            }
            if (float.IsNaN(newVelocity.magnitude))
            {
                Debug.Log("Particle " + i + " had NaN velocity " + newVelocity);
                newVelocity = Vector3.zero;
            }

            // Periodic boundary condition
            if (usePeriodicBoundary && !particleBounds.Contains(newPosition))
            {
                // Extra distance to move particle inside cube to make sure it doesn't wrap again right away.
                //  Experimentally found 0.15 to be about the smallest stable distance
                float insideOffset = 0.15f;

                // Check if outside bounds on each of the sides
                Vector3 posOffsets = particleBounds.center + particleBounds.extents;
                Vector3 negOffsets = particleBounds.center - particleBounds.extents;
                if (newPosition.x > posOffsets.x)
                    newPosition.x = -particleBounds.extents.x + insideOffset;

                else if (newPosition.x < negOffsets.x)
                    newPosition.x = +particleBounds.extents.x - insideOffset;

                else if (newPosition.y > posOffsets.y)
                    newPosition.y = -particleBounds.extents.y + insideOffset;

                else if (newPosition.y < negOffsets.y)
                    newPosition.y = +particleBounds.extents.y - insideOffset;

                else if (newPosition.z > posOffsets.z)
                    newPosition.z = -particleBounds.extents.z + insideOffset;

                else if (newPosition.z < negOffsets.z)
                    newPosition.z = +particleBounds.extents.z - insideOffset;

                //newPosition = particleBounds.ClosestPoint(newPosition);

                // Find the closest position on the bounds and stick the particle on the OPPOSITE side of the boundary (with the same velocity)
                //Vector3 closestPointOnBounds = particleBounds.ClosestPoint(newPosition);

                // Difference between new position and the closest point on the boundary surface; this vector shows the direction the particle
                //  will be moved 
                //Vector3 dir = newPosition - closestPointOnBounds;
            }

            // Update particle's position and velocity from the gpu data
            if (selectedDebugConfiguration == 0)
            {
                particles[i].position = newPosition;
                particles[i].velocity = newVelocity;
            }
            else
            {
                // Using debug configuration, so don't update fixed particles
                if (currentDebugConfiguration.HasValue && !currentDebugConfiguration.Value.particles[i].fixInSpace)
                {
                    particles[i].position = newPosition;
                    particles[i].velocity = newVelocity;
                }
            }
        }
        particleSystem.SetParticles(particles);
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
        SPHComputeShader.SetBuffer(id, densityBufID, densityBuf);

        SPHComputeShader.Dispatch(id, numThreadGroups, 1, 1);
    }

    // Compute pressure gradient, velocity diffusion, and temperature diffusion. Doing them together because their calculations
    //  are independent of each other and there are several calculations that can be reused. Includes calculation for pressure.
    //
    // PG:
    //  Inputs: particle position (for kernel grad.), pressure, density,
    //      _PressureStiffness, _ReferenceDensity, _NumParticles, _ParticleMass
    //  Outputs: PG buf
    //
    // VL:
    //  Inputs: particle position (for kernel lapl.), velocity, density, _Viscosity, _NumParticles, _ParticleMass
    //  Outputs: VL buf
    //
    // Temperature:
    //  Inputs: particle velocity, pressure, density, mass
    //  Outputs: particle temperature
    void DispatchPGandVL()
    {
        int id = SPHComputeShader.FindKernel("ComputePGandVL");
        // input - common
        SPHComputeShader.SetBuffer(id, dataInputBufID, particleDataInputBuffer);
        SPHComputeShader.SetBuffer(id, densityBufID, densityBuf);
        SPHComputeShader.SetInt(numParticlesID, numParticles);
        SPHComputeShader.SetFloat(particleMassID, particleMass);
        // input for PG
        SPHComputeShader.SetFloat(pressureStiffnessID, pressureStiffness);
        SPHComputeShader.SetFloat(referenceDensityID, referenceDensity);
        // input for VL
        SPHComputeShader.SetFloat(viscosityID, viscosity);

        // output
        SPHComputeShader.SetBuffer(id, pressureGradientBufID, pressureGradientBuf);
        SPHComputeShader.SetBuffer(id, diffusionBufID, diffusionBuf);
        SPHComputeShader.SetBuffer(id, dataOutputBufID, particleDataOutputBuffer);

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
        SPHComputeShader.SetBuffer(id, pressureGradientBufID, pressureGradientBuf);
        SPHComputeShader.SetBuffer(id, diffusionBufID, diffusionBuf);
        SPHComputeShader.SetVector(extAccelerationsID, externalAccelerations);
        SPHComputeShader.SetFloat(particleMassID, particleMass);
        //TODO: timestep!! Should be constant for CFL??? (or have cfl adjust per frame?)
        //  But for physics simulation, should be constant! Or else it's dependent on framerate
        SPHComputeShader.SetFloat(deltaTimeID, 0.001f);

        // output
        SPHComputeShader.SetBuffer(id, dataOutputBufID, particleDataOutputBuffer);

        SPHComputeShader.Dispatch(id, numThreadGroups, 1, 1);
    }
}


// This works with [numthreads(8, 8, 1)]
//SPHComputeShader.Dispatch(0, tex.width / 8, tex.height / 8, 1);

// This works with [numthreads(64, 1, 1)]
//SPHComputeShader.Dispatch(0, tex.width*4, 1, 1);

// To process x items, with a kernel group size of ThreadGroupSize*1*1, call Dispatch(x/ThreadGroupSize, 1, 1)!!
//SPHComputeShader.Dispatch(0, 256*256 / ThreadGroupSize, 1, 1);
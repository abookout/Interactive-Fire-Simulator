using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using System.Linq;
using System.IO;

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
    const int MaxNumParticles = 32768;      // 2^15 = 32768

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
        ambientTemperatureID = Shader.PropertyToID("_AmbientTemperature"),
        heatDiffusionID = Shader.PropertyToID("_HeatDiffusionRate"),
        extAccelerationsID = Shader.PropertyToID("_ExternalAccelerations");

    // Stores the actual particle positions and velocities. Array is saved over updates so it doesn't need keep being allocated
    ParticleSystem.Particle[] particlesArr;

    // Stores the actual particle temperatures. It's a Vector4 because of how the Unity particle system needs custom particle data. Only the first component is used. 
    List<Vector4> particleTemperatureArr = new List<Vector4>();

    Material particleMaterial;

    public bool showFPS = false;

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
    [SerializeField] bool resetTemperatureOnPassingPeriodicBoundary = true;
    [SerializeField] float spawnVolumeWidth = 2f;
    [SerializeField] Bounds particleBounds;

    [Header("Temperature attributes")]
    [SerializeField] Color flameParticleColor;
    [SerializeField] Color coolParticleColor;
    [SerializeField] bool colorByHeat;
    [SerializeField] float flameTempThreshold = 100;    // Min temp for particle to be a part of fire (and be rendered)

    [Header("Heat source attributes")]
    // The heat source instantly sets any particles within to the set temperature
    [SerializeField] bool useHeatSource = false;
    [SerializeField, Min(0)] float heatSourceTemperature;
    [SerializeField] Bounds heatSourceBounds;

    [Header("Buffer & shader object properties")]
    // Main compute shader for calulating update to particles
    [SerializeField] ComputeShader SPHComputeShader;


    //  Make sure there's at least one thread group!
    int numThreadGroups;

    struct ParticleDataIn
    {
        public Vector3 position;
        public Vector3 velocity;

	    // interim calculation data
        public Vector3 _pressureGradient;
        public Vector3 _diffusion;
        public float _density;
        public float _temperatureDiffusion;
        
	    // HLSL wants float3s before floats. This is not interim data
        public float temperature;
    }
    struct ParticleDataOut {
        public Vector3 position;
        public Vector3 velocity;
        public float temperature;
    }
    const int sizeofParticleDataIn = sizeof(float) * 15;
    const int sizeofParticleDataOut = sizeof(float) * 7;

    // Particle positions and current velocities are given to the CS kernels as an input.
    //  At the end, position and velocity are set by the ComputePosAndVel kernel by integration.
    [SerializeField] ComputeBuffer particleDataInputBuffer;
    // Buffer for the calculation results
    [SerializeField] ComputeBuffer particleDataOutputBuffer;

    public void RespawnParticles()
    {
        // Clear and repopulate particles
        particleSystem.Clear();

        // If using a debug config, only respawn from that
        if (selectedDebugConfiguration != 0)
        {
            DebugConfiguration config = currentDebugConfiguration.Value;
            // If using a debug configuration don't respawn as normal, just populate from configuration
            //PopulateParticleArrays();
            numParticles = config.particles.Count;
            particlesArr = new ParticleSystem.Particle[numParticles];
            particleTemperatureArr = new List<Vector4>();
            for (int i = 0; i < numParticles; i++)
            {
                DebugParticleData p = config.particles[i];
                particlesArr[i].position = p.position;
                particlesArr[i].velocity = p.velocity;
                particleTemperatureArr.Add(new Vector4(p.temperature, 0, 0, 0));
            }

            //particleSystem.SetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);
            //particleSystem.SetParticles(particlesArr, numParticles);
            return;
        }

        UpdateParticleCount();

        // Reset temperatures
        particleTemperatureArr = new List<Vector4>(Enumerable.Repeat(new Vector4(ambientTemperature, 0, 0, 0), particlesArr.Length));

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
        Gizmos.DrawWireCube(particleSystem.transform.position, particleBounds.size);

        if (useHeatSource)
        {
            Gizmos.color = Color.magenta;
            Gizmos.DrawWireCube(heatSourceBounds.center, heatSourceBounds.size);
        }
    }


    // Using OnEnable instead of start or awake so that the buffers are refreshed every hot reload
    private void OnEnable()
    {
        //Unity.Collections.LowLevel.Unsafe.UnsafeUtility.SetLeakDetectionMode(NativeLeakDetectionMode.EnabledWithStackTrace);
        particleMaterial = particleSystem.GetComponent<Renderer>().material;
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


    readonly float fpsDrawInterval = 0.1f;
    int fpsLastVal = -1;
    float fpsLastDrawTime = 0;
    private void OnGUI()
    {
        if (showFPS)
        {
            GUIStyle style = new GUIStyle();
            style.fontSize = 20;
            style.normal.textColor = Color.white;
            GUI.TextField(new Rect(5, 5, 20, 100), "FPS " + fpsLastVal, style);
        }
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

    void DebugPrintParticleData(int num)
    {
        // TODO: This stuff is just for testing
        ParticleSystem.Particle[] ps = new ParticleSystem.Particle[num];
        List<Vector4> temp = new List<Vector4>();
        particleSystem.GetParticles(ps);
        particleSystem.GetCustomParticleData(temp, ParticleSystemCustomData.Custom1);

        ParticleDataIn[] pd = new ParticleDataIn[num];
        particleDataInputBuffer.GetData(pd);
        
        const int maxNumToPrint = 10;

        Debug.Log("============== Start debug print ==============");
        for (int i = 0; i < Mathf.Min(num, maxNumToPrint); i++)
        {
            Debug.Log("Particle " + i + ":\n\tPosition: " + ps[i].position
                + "\tVelocity: " + ps[i].velocity
                + "\tTemp: " + temp[i].x
                + "\tDensity: " + pd[i]._density
                + "\n\tPressure Gradient: " + pd[i]._pressureGradient
                + "\t\tDiffusion: " + pd[i]._diffusion
                + "\n\tTemperature Diffusion: " + pd[i]._temperatureDiffusion);
        }
        Debug.Log("=============== End debug print ===============");
    }


    // Populate particle position, velocity, and temperature data from the particle system.
    // Caution: particle data only seems to be updated within the particle system after every frame - setting particle data and then 
    //  getting it again in the same update overwrites the data with the data from last frame!!!
    void PopulateParticleArrays()
    {
        particleSystem.GetParticles(particlesArr, numParticles);
        // Ensure that the temperature array is long enough (without calling GetCustomParticleData...)
        if (particleTemperatureArr.Count < numParticles)
        {
            for (int i = 0; i < numParticles; i++)
            {
                if (i >= particleTemperatureArr.Count)
                {
                    particleTemperatureArr.Add(new Vector4(ambientTemperature, 0, 0, 0));
                }
            }
        }
        //particleSystem.GetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);
    }

    void UpdateParticleCount()
    {
        particlesArr = new ParticleSystem.Particle[numParticles];
        // It has changed, so need to either destroy some or instantiate more
        if (particleSystem.particleCount > numParticles)
        {
            int oldNumParticles = particleSystem.particleCount;
            // Only set numParticles back into the array (which is less than before)
            particleSystem.GetParticles(particlesArr);
            particleSystem.SetParticles(particlesArr, numParticles);
            particleTemperatureArr.RemoveRange(numParticles, oldNumParticles - numParticles);
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

    [System.Serializable]
    public class PerfData
    {
        public int numParticles;
        public float fps;
        public PerfData(int p, float f)
        {
            numParticles = p; fps = f;
        }
    }
    [System.Serializable]
    public class PerfWrapper
    {
        public PerfData[] data;
        public PerfWrapper(PerfData[] d) { data = d; }
    }

    public void DoBenchmark()
    {
        StartCoroutine(CheckPerformance());
    }

    public IEnumerator CheckPerformance()
    {
        List<PerfData> results = new List<PerfData>();
        //int maxNumParticles = 30000;
        int increment = 500;
        float settleTime = 2f;
        Debug.Log("Benchmarking...");

        float lastFPSval = float.PositiveInfinity;
        int curNumParticles = 0;
        while(lastFPSval > 10)//(curNumParticles < maxNumParticles)
        {
            // set sim num particles
            numParticles = curNumParticles;

            yield return new WaitForSecondsRealtime(settleTime);
            float fps = GetAverageFPS();
            results.Add(new PerfData(curNumParticles, fps));

            Debug.Log(curNumParticles + ", " + fps);

            lastFPSval = fps;
            curNumParticles += increment;
        } 

        Debug.Log("Done");
        string jsonStr = JsonUtility.ToJson(new PerfWrapper(results.ToArray()));
        Debug.Log(jsonStr);
    }

    List<float> fpsSamples = new List<float>();
    int fpsMaxSamples = 20;
    public float GetAverageFPS()
    {
        float avg = 0;
        foreach (float sample in fpsSamples)
        {
            avg += sample;
        }
        avg /= fpsSamples.Count;

        return avg;
    }

    bool isFirstUpdate = true;
    private void Update()
    {
        if (Time.time > fpsLastDrawTime + fpsDrawInterval)
        {
            fpsLastVal = (int)(1 / Time.smoothDeltaTime);
            fpsLastDrawTime = Time.time;

            fpsSamples.Add(fpsLastVal);     //add to end
            if (fpsSamples.Count > fpsMaxSamples) 
                fpsSamples.RemoveAt(0);
        }

        // Make sure the number of particles hasn't changed.
        // If using a debug configuration, don't update particle count as normal
        if (selectedDebugConfiguration == 0 && particleSystem.particleCount != numParticles)
        {
            UpdateParticleCount();
        }

        if (isFirstUpdate)
        {
            // Skip update on first frame just for making gpu debugging nicer (it's hard to capture the first frame in RenderDoc).
            //  Note that removing this could cause an issue with the particle data from initialize not getting set... maybe.
            isFirstUpdate = false;
            return;
        }

        // Invariant: at the start of the update (before any logic happens), ensure that the particle data is populated from the particle system.
        //  This is to make sure we don't need to populate it again in the meanwhile, since it would overwrite anything we've written to it
        //  before the update is done.
        PopulateParticleArrays();

        // Send positions and velocities to GPU
        WriteParticleDataToGPU();


        // Perform SPH calculations
        SimulateParticles();

        // If particles are in the heat source, override their temp from it
        if (useHeatSource)
        {
            ApplyTemperatureFromHeatSource();
        }

        // Set data in particle systems to draw updated data
        particleSystem.SetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);
        particleSystem.SetParticles(particlesArr);

        //DebugPrintParticleData(numParticles);
    }

    // Initialize particle system and buffers for GPU
    void Initialize()
    {
        // note: SubUpdates prevents RenderDoc from viewing the buffer contents

        // Allocate the maximum amount of particles so that we don't need to keep creating new buffers when the number of particles changes
        particleDataInputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleDataIn);
        particleDataOutputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleDataOut);

        // Initialize buffers, using unity's particle system emission shape to choose random initial positions in a box
        //var psShape = particleSystem.shape;
        var psMain = particleSystem.main;
        psMain.maxParticles = MaxNumParticles;

        particleMaterial.color = coolParticleColor;

        var psShape = particleSystem.shape;
        psShape.scale = new Vector3(spawnVolumeWidth, spawnVolumeWidth, spawnVolumeWidth);

        // Set as constant color (doesn't change over lifetime)
        var psColor = particleSystem.colorOverLifetime;
        psColor.color = new ParticleSystem.MinMaxGradient(coolParticleColor);
        psColor.enabled = true;

        // Enable custom data for tracking temperature
        var psCustomData = particleSystem.customData;
        psCustomData.enabled = true;
        psCustomData.SetMode(ParticleSystemCustomData.Custom1, ParticleSystemCustomDataMode.Vector);
        psCustomData.SetVectorComponentCount(ParticleSystemCustomData.Custom1, 1);

        // Spawn in the initial particles
        particlesArr = new ParticleSystem.Particle[numParticles];

        particleSystem.Emit(numParticles);
        PopulateParticleArrays();

        // Evenly space them
        if (spawnEvenlySpaced)
        {
            EvenlySpaceParticles();
        }

        ParticleDataIn[] particleDataInArr = new ParticleDataIn[MaxNumParticles];

        for (int i = 0; i < MaxNumParticles; i++)
        {
            if (i < numParticles)
            {
                particleDataInArr[i].position = particlesArr[i].position;

                particleDataInArr[i].temperature = ambientTemperature;
                particleTemperatureArr[i] = new Vector4(ambientTemperature, 0, 0, 0);

                // The rest of the fields will be zeroed
            }
            else
            {
                particleDataInArr[i].position = Vector3.zero;
                // don't need to set temperature to 0 here b/c it's a list and already covered up to numParticles
            }

            particleDataInArr[i].velocity = Vector3.zero;
        }

        particleSystem.SetParticles(particlesArr);
        particleSystem.SetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);

        particleDataInputBuffer.SetData(particleDataInArr);
        particleDataOutputBuffer.SetData(new ParticleDataOut[MaxNumParticles]);       // Zero-initialize output buffer
    }

    // Initialize as normal except populate particle data from the selected debug configuration
    void InitializeFromDebugConfig()
    {
        // Allocate the maximum amount of particles so that we don't need to keep creating new buffers when the number of particles changes
        particleDataInputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleDataIn);
        particleDataOutputBuffer = new ComputeBuffer(MaxNumParticles, sizeofParticleDataOut);

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
        //particleSystem.Emit(numParticles);

        // Enable custom data for tracking temperature
        var psCustomData = particleSystem.customData;
        psCustomData.enabled = true;
        psCustomData.SetMode(ParticleSystemCustomData.Custom1, ParticleSystemCustomDataMode.Vector);
        psCustomData.SetVectorComponentCount(ParticleSystemCustomData.Custom1, 1);

        //TODO: testing
        //particleSystem.GetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);
        particleTemperatureArr = new List<Vector4>();

        ParticleDataIn[] particleDataInArr = new ParticleDataIn[MaxNumParticles];
        for (int i = 0; i < MaxNumParticles; i++)
        {
            if (i < numParticles)
            {
                DebugParticleData particle = curDebugConfig.particles[i];
                particleDataInArr[i].position = particle.position;
                particleDataInArr[i].velocity = particle.velocity;
                particleDataInArr[i].temperature = particle.temperature;
                // Populate data in ParticleSystem so update uses that
                particlesArr[i].position = particle.position;
                particlesArr[i].velocity = particle.velocity;
                particleTemperatureArr.Add(new Vector4(particle.temperature, 0, 0, 0));
            }
            else
            {
                particleDataInArr[i].position = Vector3.zero;
                particleDataInArr[i].velocity = Vector3.zero;
                // don't need to set temperature to 0 here b/c it's a list and already covered up to numParticles
            }
        }

        particleSystem.SetParticles(particlesArr);
        particleSystem.SetCustomParticleData(particleTemperatureArr, ParticleSystemCustomData.Custom1);

        particleDataInputBuffer.SetData(particleDataInArr);
        particleDataOutputBuffer.SetData(new ParticleDataOut[MaxNumParticles]);       // Zero-initialize output buffer
    }

    void ReleaseBuffers()
    {
        particleDataInputBuffer.Release();
        particleDataOutputBuffer.Release();
    }

    // Write current particle positions and velocities for acceleration CS
    void WriteParticleDataToGPU()
    {
        //TODO: maintain position and velocity arrays so don't need to access from particle system? Not sure if this is a problem
        ParticleDataIn[] dataArray = new ParticleDataIn[numParticles];
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

        //particleSystem.SetParticles(particlesArr, numParticles);
    }

    // Set any particles within the heat source to its temperature
    void ApplyTemperatureFromHeatSource()
    {
        for (int i = 0; i < particlesArr.Length; i++)
        {
            if (heatSourceBounds.Contains(particlesArr[i].position))
            {
                particleTemperatureArr[i] = new Vector4(heatSourceTemperature, 0, 0, 0);
            }
        }
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

        //// Position, velocity, and temp buffer now updated, so set particles from that.
        ParticleDataOut[] dataArray = new ParticleDataOut[numParticles];
        particleDataOutputBuffer.GetData(dataArray);

        // Update particle position, velocity, use boundary conditions, apply coloring based on temperature
        for (int i = 0; i < numParticles; i++)
        {
            Vector3 newPosition = dataArray[i].position;
            Vector3 newVelocity = dataArray[i].velocity;
            float newTemperature = dataArray[i].temperature;

            ////// TODO: Testing: if particle got NaN'd, fix it so it's easier to find the problem
            ///
            /*
            if (float.IsNaN(newPosition.magnitude))
            {
                Debug.LogWarning("Particle " + i + " had NaN position " + newPosition);
                newPosition = Vector3.zero;
            }
            if (float.IsNaN(newVelocity.magnitude))
            {
                Debug.LogWarning("Particle " + i + " had NaN velocity " + newVelocity);
                newVelocity = Vector3.zero;
            }
            */

            bool particleFixed = currentDebugConfiguration.HasValue && currentDebugConfiguration.Value.particles[i].fixInSpace;

            // Periodic boundary condition
            if (usePeriodicBoundary && !particleBounds.Contains(newPosition) && !particleFixed)
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

                // Resetting the temperature when a particle jumps across to the other side of the boundary simulates ambient-temperature air flowing in.
                //  Acts as a heat sink.
                if (resetTemperatureOnPassingPeriodicBoundary)
                {
                    newTemperature = ambientTemperature;
                }
            }

            // Update particles from the gpu data
            if (selectedDebugConfiguration == 0)
            {
                particlesArr[i].position = newPosition;
                particlesArr[i].velocity = newVelocity;
                particleTemperatureArr[i] = new Vector4(newTemperature, 0, 0, 0);
            }
            // Using debug configuration, so don't update fixed particles' positions in space
            else if (currentDebugConfiguration.HasValue)
            {
                // Update temperature regardless
                particleTemperatureArr[i] = new Vector4(newTemperature, 0, 0, 0);
                if (!particleFixed)
                {
                    particlesArr[i].position = newPosition;
                    particlesArr[i].velocity = newVelocity;
                }
            }

            // Color based on temp
            //TODO make this not ugly
            particlesArr[i].startColor = !colorByHeat ? flameParticleColor : (newTemperature > flameTempThreshold ? flameParticleColor : coolParticleColor);
        }
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

        SPHComputeShader.Dispatch(id, numThreadGroups, 1, 1);
    }

    //TODO: part of comment may be wrong
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
        SPHComputeShader.SetInt(numParticlesID, numParticles);
        SPHComputeShader.SetFloat(particleMassID, particleMass);
        // input for PG
        SPHComputeShader.SetFloat(pressureStiffnessID, pressureStiffness);
        SPHComputeShader.SetFloat(referenceDensityID, referenceDensity);
        // input for VL
        SPHComputeShader.SetFloat(viscosityID, viscosity);
        // input for temp diffusion
        SPHComputeShader.SetFloat(heatDiffusionID, temperatureDiffusionRate);

        // output
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

        SPHComputeShader.SetFloat(ambientTemperatureID, ambientTemperature);
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
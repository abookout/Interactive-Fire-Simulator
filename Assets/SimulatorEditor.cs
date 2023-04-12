using UnityEngine;
using UnityEditor;
using System.Linq;

[CustomEditor(typeof(FireSim))]
public class SimulatorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        FireSim fs = (FireSim)target;

        GUILayout.Label("Debug controls");

        if (GUILayout.Button("Respawn Particles"))
        {
            fs.RespawnParticles();
        }

        if (GUILayout.Button("Start FPS benchmark"))
        {
            fs.DoBenchmark();
        }

        GUILayout.Label("Select debug configuration");
        // Select labels and prepend "None"
        string[] options = fs.debugConfigurations.Select(x => x.description).Prepend("None").ToArray();
        fs.selectedDebugConfiguration = GUILayout.SelectionGrid(fs.selectedDebugConfiguration, options, fs.debugConfigurations.Count+1);

        GUILayout.Space(20);
        DrawDefaultInspector();
    }
}
using UnityEngine;
using UnityEditor;

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

        //GUILayout.SelectionGrid(0, new string[] { "a", "b", "c", "d", "e" }, 6);

        GUILayout.Space(20);
        DrawDefaultInspector();
    }
}
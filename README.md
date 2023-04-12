This is a real-time particle-based fire simulation using SPH. It's implemented in a compute shader on the GPU using HLSL with Unity.

## Demos

These example videos are simulating 4096 particles at ~30-40 fps on a GTX 970 <b>without nearest neighbor search.</b> It's pretty much as simple as possible for a proof of concept fire sim. It uses an unsupported periodic boundary and naively renders the particles using Unity's builtin particle system. The particle system is only used for rendering and random emission (within a Bounds) when particles are added.

https://user-images.githubusercontent.com/19939886/231599111-782e40d5-d33d-4ba2-a225-bfc00d247fd1.mp4

https://user-images.githubusercontent.com/19939886/231599128-2692ed6c-81a2-4b3f-939b-276c4c5ee0d2.mp4


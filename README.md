### Steps to Run:

While setting up the environment, make sure to uncomment any lines that have been commented out. Installing the dependencies may take some time, so please be patient during the process.

### What You Can Do

You can generate a 3D model using either of the following methods:

1. **From Your Own Images:** Upload a 2D image which TripoSR will use to create a 3D representation.

2. **From Text Description (Prompt):**  
   - Provide a natural language description of the object.
   - The prompt is sent to **Stable Diffusion** by StabilityAI to generate a realistic 2D image.
   - The generated image is then passed to **TripoSR**, which reconstructs a 3D object from it.

### Tools Used
- **TripoSR**: For 3D surface reconstruction from single/multiple views.
- **Stable Diffusion**: For generating synthetic 2D images from text prompts.

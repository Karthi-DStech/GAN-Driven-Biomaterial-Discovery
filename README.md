<h2>Welcome to [AI-based Advancements for Biomaterial Discovery: Comparative Analysis of GAN Models in Topography Generation]</h2>

<h3>Thank you for visiting the repository of [GANs-for-BIO-MATERIAL-DISCOVERY]! This project is dedicated to exploring the innovative application of Generative Adversarial Networks (GANs) for Bio-Material Discovery.</h3>

<h3>This Repository includes detailed documentation, source code, and results from our research.</h3>


**IN THIS READ ME FILE, USERS CAN FIND STEP BY STEP INSTRUCTIONS AND FEEL FREE TO EXPLORE.**

- In the **IEEE Research Paper GANs folder**, users can find the research paper. 



- In the files of **Vanilla GANs, C-GANs, DC-GANs and WAN**, users can find all the coding Notebooks and each notebook files is named according to the specific GAN combination it represents.



- For the **GANs-generated topographic images**, navigate to the "GANs - Generated Topographic Images" folder. Within this directory, you will find subfolders named after each specific GAN variant. For further information, check the tree map below.

 <img width="275" alt="Screenshot 2024-01-31 at 09 57 21" src="https://github.com/Karthi-DStech/GANs-for-BIO-MATERIAL-DISCOVERY/assets/126179797/f4cfb293-c511-4c82-938f-79117db3d0e3">


- In the **Preprocessing Coding Files directory**, users will find a collection of Jupyter notebooks designed to handle various stages of image processing and analysis for biomaterial data. For further information, check the below tree map.

<img width="716" alt="Screenshot 2024-01-31 at 09 57 55" src="https://github.com/Karthi-DStech/GANs-for-BIO-MATERIAL-DISCOVERY/assets/126179797/6b96c405-aff7-4f96-8d25-64bf97fb8995">



- In the **Preprocessed for FID Score** folder, users will find images that have been cropped to their specific sizes in line with the requirements for calculating the Fréchet Inception Distance (FID) Score. Each folder within is named to correspond with the specific GAN combination used in the coding notebooks.



**INSTRUCTIONS FOR ACTIVATING TENSORBOARD**

- Guidelines for utilizing TensorBoard on a laptop or HPC are as follows:
- Ensure TensorBoard is installed.
- The summary writer will generate a directory named 'logs' or 'runs' within the same location as the notebook.
- Navigate to the directory containing the TensorBoard logs using the Command Line Interface (CLI), for example, cd [path_to_directory].
- Execute the following command, modifying it to match the name of your generated summary file.

tensorboard --logdir=runs

- A link will be displayed in the command-line interface. Click on it to open the tensorboard in your browser or copy and paste the link.
- If you are using Google Colab, Just run the notebook and the tensorboard will be activated automatically. 


**Other Instructions**

- Ensure you have all the required libraries installed before running the notebooks.
- Modify parameters and paths as needed to fit your specific dataset and requirements.
- Make sure all the notebooks and the data folder are in same Directory.
- Each dataset is zipped for compact storage and needs to be uncompressed before use.

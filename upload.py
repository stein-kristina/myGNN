from huggingface_hub import login, upload_folder, upload_file

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
# upload_folder(folder_path="myGNN.zip", repo_id="Amadeus10086/mygnn", repo_type="model")
upload_file(path_or_fileobj="myGNN.zip", path_in_repo="myGNN.zip", repo_id="Amadeus10086/mygnn", repo_type="model")
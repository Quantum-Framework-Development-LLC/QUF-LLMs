from azureml.core import Workspace, Model

def deploy_model():
    ws = Workspace.from_config()

    model = Model.register(workspace=ws,
                           model_name='GPT2-Model',
                           model_path='./output', # local path
                           description='GPT2 Language Model')

    # Your Azure deployment code here

if __name__ == "__main__":
    deploy_model()

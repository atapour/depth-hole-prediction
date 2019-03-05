from model.the_model import TheModel

def create_model(args):
    model = TheModel()
    model.initialize(args)
    print("The model has been created.")
    return model

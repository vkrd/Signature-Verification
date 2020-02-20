from helper_functions import *

# Custom data loader
load_data()

# Load network and model
network = create_network()
model = create_model(network)
model.compile(Adam())

model.summary()

print("Training starting...\n")

epochs = 50
start = time.time()

model_updates = 5

for i in range(epochs):
    pass
    triplets = get_mixed_batch(64, 16, 16, network)
    loss = model.train_on_batch(triplets, None)
    if i % model_updates == 0:
        print("Epoch: " + str(i+1) + "/" + str(epochs) + " Time Lapsed: " + str(time.time()-start) + " seconds Loss: " + str(loss))

print("Training complete")
from helper_functions import *

# Custom data loader
load_data()

# Load network and model
network = create_network()
model = create_model(network)
model.compile(Adam())

# Uncomment to find out best input shape for model (default best is [275, 656])
# print(calculate_average_shape())

print("Training starting...\n")

epochs = 10
start = time.time()
prev = start

model_updates = 1

for i in range(epochs):
    triplets = get_mixed_batch(64, 16, 16, network)
    loss = model.train_on_batch(triplets, None)
    if i % model_updates == 0:
        print("Epoch: " + str(i+1) + "/" + str(epochs) + " | Time Lapsed: " + str(round(time.time()-prev, 1)) + " seconds | Loss: " + str(loss))
    prev = time.time()

print("Training complete\n")

save_model(model)

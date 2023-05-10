import openml
import pickle

for d in ["mnist_784", "a9a", "vehicleNorm", "webdata_wXa", "sylva_prior", "jasmine", "madeline", "philippine", "musk", "SantanderCustomerSatisfaction"]:
    data = openml.datasets.get_dataset(d)
    x, y, _, _ = data.get_data(target=data.default_target_attribute)
    print(x.shape, y.shape)
    with open(f'data/{d}.pkl', 'wb') as handle:
        pickle.dump({"x": x, "y": y}, handle, protocol=4)


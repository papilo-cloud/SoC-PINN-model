from train import train
from visualize import visualize

CSV_PATH = 'battery_data_pinn.csv'
BATTERY_CAPACITY = 2.3 # in Ah

if __name__ == "__main__":
    model = train(CSV_PATH, BATTERY_CAPACITY)
    visualize(model, CSV_PATH, BATTERY_CAPACITY)
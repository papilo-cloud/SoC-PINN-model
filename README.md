# Physics-Informed Neural Network (PINN) for State of Charge (SoC) Estimation
This project implements an unsupervised Physics-Informed Neural Network (PINN) to estimate the State of Charge (SoC) of a battery using only current and time data. No SoC labels are required — the model learns by enforcing physical laws during training.

### Sample Data Format
<pre> 
time, current
0,1.0
1,1.0
2,1.0
</pre>

- `time`: in seconds

- `current`: in Amperes (positive = discharge)

### Model Architecture
- `Input`: Time 𝑡 (in hours)

- `Output`: Estimated SoC

- `Hidden Layers`: 2 layers with 64 neurons, `Tanh` activation

- `Loss`: Physics loss from battery ODE

### Physics Constraint (Battery Equation)
    $\frac{dSoC}{dt} = - \frac{I(t)}{\cdot{3600}{C}}$

### How to Run
1. Install dependencies
    pip install torch pandas matplotlib
2. Place your battery CSV data in the project directory
    Example: battery_data_pinn.csv

3. Run the training and plotting pipeline
    python demo.py
    This will:

    - Train the PINN model

    - Plot estimated SoC vs Coulomb counting reference

### Why Use PINNs?
- Requires **no labeled SoC data**
- Incorporates **domain knowledge** into ML model
- Works well with **small datasets**
- Enables learning from **first principles**

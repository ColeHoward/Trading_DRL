# Instructions

1. Clone this repository
2. Make sure you have Docker installed and running
3. Build the image with `docker build -t <image_name> .`
4. Run the container with `docker run -v $(pwd):/app -it <image_name> /bin/bash`
5. Inside the container, run `python3 main.py <stock_ticker>` to run the training script
    - Expect training to take over 3 hours on a CPU, 75 minutes on a T4 GPU, and 60 minutes on a V100 GPU
6. If you want to try the original model, you have to change the model and model args in main.py
7. After training, test and training results will be automatically written to the `results` folder, at which point you can run analyze.py to generate graphs and metrics
   - you'll have to modify the original arguments of analyze.py depending which stock(s) you choose for training



# Energy-Consumption-Forecasting
By leveraging state-of-the-art deep learning techniques this project aims at predicting energy consumption for PJM, a major energy company in the United States. ​

## Why it matters?
Currently, 85% of energy is provided through the use of non-renewable energy sources. Thus producing extensive amounts of carbon dioxide, influencing the environment negatively. This carbon dioxide effectively captures the ultra violet light, keeping it within the atmosphere and increasing heat. This leads into many further impacts such as wildfires, droughts, habitat loss, etc. All of which ultimately impact humanity, and our ecosystems posing a concerning issue. ​

However, recent ongoing developments, such as renewable energy sources have proven themselves to be commendable, and capable of supporting societies, but it is hard to get these new methods running, and effecrtively implement them without risk Subsequently, through the use of artificial intelligence, future energy consumption values can be predicted. ​

Providing governments, and officials to plan the integration of renewable energy sources, facilitating their job, and removing any risk of failure, considering that they know exactly how much energy the country will need, all of which contribute towards clean energy, positively impacting climate change, and contributing towards a more sustainable world, linking with SDG7 and SDG13​

## Results
In order to test the success of my unique architecture, I developed the model through python and PyTorch, both of which are popular choices in machine learning. I have decided to test it on training data with only 5000 data points in order to effectively test its ability to capture complex patterns with little data. Furthermore, I trained a separate model on the full extent of the training data to make predictions on values that it has never seen before, providing it with the opportunity to showcase how it applies its learned patterns to real non-seen data, serving as a more accurate indicator of the precision of the model. (As evident the model does exceptionally well when forecasting both training data and testing data, proving the reliability of the model.)​

<img width="400" height="100" alt="Screenshot 2025-01-27 at 00 08 42" src="https://github.com/user-attachments/assets/f865a7c1-bfb2-46d9-a018-50297d596b47" />
<img width="400" height="100" alt="Screenshot 2025-01-27 at 00 09 30" src="https://github.com/user-attachments/assets/cf9cd434-2726-485d-b36b-007a54ef9820" />

## How it works
​Essentially, this model works through a modification of the decoder transformer architecture (GPT) originally introduced through machine learning leaders; OpenAI. However, this model was typically applied against language modelling, and could overfit simpler data, like energy consumption. Subsequently, I have changed the model into a more unique architecture, adapted to apply to this specific dataset. ​

This unique architecture had two multi-perceptron layers, one attention mechanism, two layer normalization, and one final linear layer. All of which are extremely simple, yet capable of achieving powerful performance.​

The first multi-perceptron layer serves as an initial processing. It has the dimensions of (100, 60, 120), all of which extract the initial sequence to a more complex detailed tensor. Then, the attention mechanism essentially gets this sequence, and runs it through a compilation of blocks that turn it into a matrix that knows the detailed relationships between each value, allowing it to capture more long-term information. Then, the multi-perceptron layer gets these attention values, and runs them through a final processing layer, that has the dimensions of (120, 80, 60).  Finally, the final linear turns this matrix into one single value, serving as the output, or the final prediction. To the bottom you will find a more detailed explanation on how every layer works below:​

## Multi-Perceptron-Layer
The multi-perceptron layer works by setting random values, and then multiplying these values by the inputs sequentially, producing a final output, then the difference is found between these, and this value is called the loss. A loss that needs to be minimized. Subsequently, we find the derivative of the loss with respect to every single variable, multiply it by a stable learning rate, and subtract this loss from the number, allowing the model to perform calculations that produce a lower loss, or more accurate results. This essentially works for the whole model, the loss is back propagated since the output of one layer is the input of another, so by finding the derivative of the input, you get the output for the last layer, and this can be used as the loss for the calculus  of the previous layer. Please find the mathematical details below, with a simple example of inputs and outputs. The diagram to the left is the forward propagation, while the one to the left is the backward propagation​

​<img width="419" alt="Screenshot 2025-01-27 at 11 29 45" src="https://github.com/user-attachments/assets/b6a0f4fb-0ab9-4da9-af60-787b43235ccf" />
<img width="419" alt="Screenshot 2025-01-27 at 11 37 41" src="https://github.com/user-attachments/assets/73e6a28a-8da5-4cae-b4e9-40860270dfa9" />

## Layer-Normalization
This complex architecture deals with many values, and layers. Subsequently, it is very prone to suffer from exploding gradients, or noisy data, in order to fix this, the model will leverage “Layer Normalization” to stabilize the values. This layer is typically employed on large language models. Find the mathematical formula for it:​

​Layer normalization works differently than typical normalization techniques. In essence, layer normalization takes the mean of the array, and subtracts every value by it. Then it divides it by the square root of the standard deviation squared plus epsilon. This value is then multiplied and added by learnable parameters, giving the layer normalization a chance to adapt to the data and model, making this commendable.​

## Attention Mechanism
The attention mechanism it one of the most critical parts of the entire model, it provides the model with the ability to capture long term relationships, helping it see how different values relate. Essentially, the attention mechanism gets the values and runs them through three different linear layers, producing the queries, keys, and values. After this the transposed keys undergo matrix multiplication with the values, creating the scores. The scores get scaled and then go through a softmax activation function. Then these processed scores undergo matrix multiplication with the values, producing an array with the same dimensions as the initial input. Finally, this array goes through a linear layer for final processing. Find it to the right​

​

 ​

​


​
​

​

​

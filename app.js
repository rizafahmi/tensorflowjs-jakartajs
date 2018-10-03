import * as tf from "@tensorflow/tfjs";
import "babel-polyfill";

// Create model
const model = tf.sequential();
// Create layers
const hidden = tf.layers.dense({
  units: 4,
  inputShape: [2],
  activation: "sigmoid"
});
const output = tf.layers.dense({
  units: 1,
  activation: "sigmoid"
});
// Add layers
model.add(hidden);
model.add(output);
// Prepare model for training
model.compile({
  optimizer: tf.train.sgd(0.1),
  loss: "meanSquaredError"
});
// Prepare the data
const inputs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
const outputs = tf.tensor2d([[0], [1], [1], [0]]);
// Train the model
const train = async () => {
  const config = { shuffle: true, epochs: 10 };
  for (let i = 0; i < 1000; i += 1) {
    const response = await model.fit(inputs, outputs, config);
    console.log(response.history.loss[0]);
  }
};
// Predict
const predict = async () => {
  await train();
  console.log("training complete");
  const prediction = model.predict(inputs);
  prediction.print();
};

predict();

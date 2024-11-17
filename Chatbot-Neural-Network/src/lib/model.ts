import * as tf from '@tensorflow/tfjs';

// Intents data for training
const intents = [
  { tag: 'greeting', patterns: ['hi', 'hello', 'hey', 'howdy'], responses: ['Hello!', 'Hi there!', 'Hey!'] },
  { tag: 'goodbye', patterns: ['bye', 'goodbye', 'see you', 'cya'], responses: ['Goodbye!', 'See you later!', 'Bye!'] },
  { tag: 'thanks', patterns: ['thanks', 'thank you', 'appreciate it'], responses: ['You\'re welcome!', 'No problem!', 'My pleasure!'] },
  { tag: 'help', patterns: ['help', 'support', 'assist'], responses: ['How can I help you?', 'What do you need help with?'] }
];

// Prepare training data
export function prepareTrainingData() {
  const words = [...new Set(intents.flatMap(intent => 
    intent.patterns.flatMap(pattern => pattern.toLowerCase().split(' '))))];
  
  const tags = [...new Set(intents.map(intent => intent.tag))];
  
  const trainingData = intents.flatMap(intent =>
    intent.patterns.map(pattern => {
      const bag = words.map(word => pattern.toLowerCase().includes(word) ? 1 : 0);
      const output = tags.map(tag => intent.tag === tag ? 1 : 0);
      return { input: bag, output };
    })
  );

  return { words, tags, trainingData };
}

// Create and train the model
export async function createAndTrainModel(trainingData: any) {
  const model = tf.sequential();
  
  model.add(tf.layers.dense({
    units: 8,
    inputShape: [trainingData[0].input.length],
    activation: 'relu'
  }));
  
  model.add(tf.layers.dense({
    units: trainingData[0].output.length,
    activation: 'softmax'
  }));
  
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  const xs = tf.tensor2d(trainingData.map((item: any) => item.input));
  const ys = tf.tensor2d(trainingData.map((item: any) => item.output));
  
  await model.fit(xs, ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs?.loss.toFixed(4)}`);
      }
    }
  });
  
  return model;
}

// Predict response
export function predictResponse(input: string, model: tf.LayersModel, words: string[], tags: string[]) {
  const inputBag = words.map(word => input.toLowerCase().includes(word) ? 1 : 0);
  const prediction = model.predict(tf.tensor2d([inputBag])) as tf.Tensor;
  const tagIndex = prediction.argMax(-1).dataSync()[0];
  const intent = intents.find(intent => intent.tag === tags[tagIndex]);
  return intent?.responses[Math.floor(Math.random() * intent.responses.length)] || "I don't understand.";
}
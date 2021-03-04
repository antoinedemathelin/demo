import * as tf from '@tensorflow/tfjs';


export function baseModelTwoLayer(C=10, lr=0.001, inputShape=1, activation='linear', init='zeros') {	
	const model = tf.sequential();
	model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [inputShape],
								kernelConstraint: tf.constraints.maxNorm({maxValue: C}),
								biasConstraint: tf.constraints.maxNorm({maxValue: C}),
								}));
	model.add(tf.layers.dense({units: 100, activation: 'relu',
								kernelConstraint: tf.constraints.maxNorm({maxValue: C}),
								biasConstraint: tf.constraints.maxNorm({maxValue: C}),
								}));
	model.add(tf.layers.dense({units: 1, activation: activation,
								kernelConstraint: tf.constraints.maxNorm({maxValue: C}),
								biasConstraint: tf.constraints.maxNorm({maxValue: C}),
								biasInitializer: init
								}));
	model.compile({loss: 'meanSquaredError', optimizer: tf.train.adam(lr)});
	return model;
};



export function baseModelOneLayer(C=1, lr=0.001, inputShape=8, activation='linear', init='zeros') {
	const model = tf.sequential();
	model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [inputShape],
								kernelConstraint: tf.constraints.maxNorm({maxValue: C}),
								biasConstraint: tf.constraints.maxNorm({maxValue: C}),
								kernelInitializer: 'glorotUniform'
								}));
	model.add(tf.layers.dense({units: 1, activation: activation,
								kernelConstraint: tf.constraints.maxNorm({maxValue: C}),
								biasConstraint: tf.constraints.maxNorm({maxValue: C}),
								kernelInitializer: 'glorotUniform',
								biasInitializer: init
								}));
	model.compile({loss: 'meanSquaredError', optimizer: tf.train.adam(lr)});
	return model;
};


export function baseModelZeroLayer(C=10, lr=0.001, inputShape=8, activation='linear', init='zeros') {
	const model = tf.sequential();
	model.add(tf.layers.dense({units: 1, activation: activation, inputShape: [inputShape],
								kernelConstraint: tf.constraints.maxNorm({maxValue: C}),
								biasConstraint: tf.constraints.maxNorm({maxValue: C}),
								biasInitializer: init
								}));
	model.compile({loss: 'meanSquaredError', optimizer: tf.train.adam(lr)});
	return model;
};
import * as tf from '@tensorflow/tfjs';

var isFitting = true;


function customLossDisc (yTrue, yPred) {
	return tf.neg(yPred);
};
function customLoss (yTrue, yPred) {
	return yPred;
};

function discrepancyNetwork (shape, baseModel, C, lr) {
	const discrepancer = baseModel(C, lr, shape);
  
	const minusOnes = tf.input({shape: [1]});
	const inputSource = tf.input({shape: [shape]});
	const inputTarget = tf.input({shape: [shape]});
	const outputSource = tf.input({shape: [1]});
	const outputTarget = tf.input({shape: [1]});
	const weightSource =  tf.input({shape: [1]});

	const outputDiscS = discrepancer.apply(inputSource);
	const outputDiscT = discrepancer.apply(inputTarget);
  
	const diffOutputDiscS = tf.layers.add().apply([outputDiscS, outputSource]);
	const diffOutputDiscT = tf.layers.add().apply([outputDiscT, outputTarget]);
	//const diffOutputDiscMinusT = tf.layers.multiply().apply([diffOutputDiscT,
	//													   minusOnes]);
	const diffOutputDiscMinusS = tf.layers.multiply().apply([diffOutputDiscS,
														   minusOnes]);
	
	const weightedDiscS = tf.layers.multiply().apply([weightSource, diffOutputDiscS]);

	//const discLossS = tf.layers.dot({axes: 0}).apply([weightedDiscS, diffOutputDiscS]);
	//const discLossT = tf.layers.dot({axes: 0}).apply([diffOutputDiscMinusT, diffOutputDiscT]);
	const discLossS = tf.layers.dot({axes: 0}).apply([weightedDiscS, diffOutputDiscMinusS]);
	const discLossT = tf.layers.dot({axes: 0}).apply([diffOutputDiscT, diffOutputDiscT]);

	const discLoss = tf.layers.add().apply([discLossS, discLossT]);
//	const discLossSquare = tf.layers.multiply().apply([discLoss, discLoss]);

	const model = tf.model({
	inputs: [inputSource, inputTarget, outputSource, outputTarget,
		   weightSource, minusOnes],
	outputs: discLoss});  //discLossSquare
	model.compile({optimizer: tf.train.adam(lr),
				 loss: customLossDisc
				 });
	return model;
};



export function createWann (shape, baseModel, baseModelDisc, C=1, Cw=1, init="zeros", lr=0.001) {

  const weightsPredictor = baseModel(Cw, lr, shape, 'relu', init);  
  const task = baseModel(C, lr, shape);
  const discrepancer = discrepancyNetwork(shape, baseModelDisc, C, lr);
  
  const minusOnes = tf.input({shape: [1]});
  const inputSource = tf.input({shape: [shape]});
  const inputTarget = tf.input({shape: [shape]});
  const outputSource = tf.input({shape: [1]});
  const outputTarget = tf.input({shape: [1]});

  const weightSource = weightsPredictor.apply(inputSource);     
  const outputTaskS = task.apply(inputSource);
  const outputTaskT = task.apply(inputTarget);

  const diffOutputTaskS = tf.layers.add().apply([outputTaskS, outputSource]);
  const diffOutputTaskT = tf.layers.add().apply([outputTaskT, outputTarget]);

  const weightedTaskS = tf.layers.multiply().apply([weightSource, diffOutputTaskS]);

  const taskLossS = tf.layers.dot({axes: 0}).apply([weightedTaskS, diffOutputTaskS]);
  const taskLossT = tf.layers.dot({axes: 0}).apply([diffOutputTaskT, diffOutputTaskT]);

  const discLoss = discrepancer.apply([inputSource, inputTarget, outputSource,
                                       outputTarget, weightSource, minusOnes]);

  const loss = tf.layers.add().apply([taskLossS, taskLossT, discLoss]);
  const model = tf.model({
    inputs: [inputSource, inputTarget, outputSource, outputTarget, minusOnes],
    outputs: loss});
  model.compile({optimizer: tf.train.adam(lr),
                 loss: customLoss
                 });
  return [model, weightsPredictor, task, discrepancer];
};



export async function trainWann (wann, weightsPredictor, task, discrepancer,
					epochs, batchSize, xSource, ySource,
                    xTarget, yTarget, dataset, xbatchTensor, constraint) {
	batchSize = parseInt(batchSize);
	epochs = parseInt(epochs);
	
	if (batchSize > ySource.shape[0]) {
		console.log("WARNING batchSize bigger than sample size");
	};
	if (yTarget.shape[0] > ySource.shape[0]) {
		console.log("WARNING target sample size bigger than source sample size");
	};
	
	var i = 1;
	
	if (dataset == "Toy"){
		var bar = document.getElementById('progress_bar');
		var percentage = document.getElementById('progress_percentage');	
	};
	if (dataset == "UCI"){
		var bar = document.getElementById('progress_bar_UCI');
		var percentage = document.getElementById('progress_percentage_UCI');	
	};
	if (dataset == "Kin"){
		var bar = document.getElementById('progress_bar_kin');
		var percentage = document.getElementById('progress_percentage_kin');	
	};
	
	var width = 1;
	
	function move() {
	  if (width >= 100) {
		i = 0;
	  } else {
		width++;
		bar.style.width = width + "%";
		percentage.innerHTML = width + "%";
	  }
	};
	
	const xsTensor = xSource;
	var saveHistoryTask = [];
	var saveHistoryWeight = [];
	saveHistoryTask.push(tf.zeros([32]).dataSync());
	saveHistoryWeight.push(weightsPredictor.predictOnBatch(xsTensor).as1D().dataSync());
	
	// shuffle
	if (dataset == "Toy"){
		var randomIndices = tf.util.createShuffledIndices(ySource.shape[0]);
		var xSourceShuffle = [];
		var ySourceShuffle = [];
		var xSourceArray = xSource.dataSync();
		var ySourceArray = ySource.dataSync();
		for (var ind of randomIndices) {
			xSourceShuffle.push(xSourceArray[ind]);
			ySourceShuffle.push(ySourceArray[ind]);
		};
		xSource = tf.tensor2d(xSourceShuffle, [ySource.shape[0], 1]);
		ySource = tf.tensor1d(ySourceShuffle);
	};

	var divLength =  Math.floor(ySource.shape / yTarget.shape) + 1;
	var xTargetComplete = xTarget.tile([divLength, 1]);
	xTargetComplete = xTargetComplete.slice([0, 0], [ySource.shape, xTarget.shape[1]]);
	var yTargetComplete = yTarget.tile([divLength]);
	yTargetComplete = yTargetComplete.slice(0, ySource.shape);
	

	var batchNbr = Math.floor(ySource.shape / batchSize);
	
	for (var i=0; i<epochs; i++) {
		
		if (i > 0) {
				var diff = parseInt(i * ( 100 / epochs)) - parseInt((i-1) * ( 100 / epochs));
				for (var j=0 ; j < diff; j++) {
					move();
				};
			};
		
		for (var j=0; j<batchNbr; j++) {
			
			if (dataset == "Toy") {
				var stopTraining = localStorage.getItem("stoptraining");
			};
			if (dataset == "UCI") {
				var stopTraining = localStorage.getItem("stoptrainingUCI");
			};
			
			if (stopTraining === 'true') {
				if (dataset == "Toy") {
					var copyHistory = [saveHistoryTask, saveHistoryWeight];
					localStorage.setItem("copyHistory", JSON.stringify(copyHistory));
				};
				return;} else {
			
			if (j == batchNbr) {
			var xBatchSource = xSource.slice([batchSize * j, 0],
										[ySource.shape - batchSize * j, xSource.shape[1]]);
			var xBatchTarget = xTargetComplete.slice([batchSize * j, 0],
										[ySource.shape - batchSize * j, xTarget.shape[1]]);
			var yBatchSource = ySource.slice(batchSize * j, ySource.shape - batchSize * j);
			var yBatchTarget = yTargetComplete.slice(batchSize * j, ySource.shape- batchSize * j);
			} else {
			var xBatchSource = xSource.slice([batchSize * j, 0],
										[batchSize, xSource.shape[1]]);
			var xBatchTarget = xTargetComplete.slice([batchSize * j, 0],
										[batchSize, xTarget.shape[1]]);
			var yBatchSource = ySource.slice(batchSize * j, batchSize);
			var yBatchTarget = yTargetComplete.slice(batchSize * j, batchSize);
			};
			var minusOnesArray = [];
			for (var k=0; k<yBatchTarget.shape; k++) {minusOnesArray.push(-1)};
			var minusOnes = tf.tensor2d(minusOnesArray, [minusOnesArray.length, 1]);
			var sourceWeights = weightsPredictor.predictOnBatch(xBatchSource);
			
			for (var layer of discrepancer.layers) {	
			if (layer.name.includes('sequential')) {
			  for (var lay of layer.layers) {
				if (! lay.name.includes('input')) {
				  lay.trainable = false;
				  if (constraint == true) {
				  lay.kernel.val.assign(lay.kernel.constraint.apply(lay.kernel.val));
				  lay.bias.val.assign(lay.bias.constraint.apply(lay.bias.val));
				  };
				  };
			  };
			};
			};
			
			await wann.trainOnBatch([xBatchSource, xBatchTarget,
								tf.neg(yBatchSource.reshapeAs(minusOnes)),
								tf.neg(yBatchTarget.reshapeAs(minusOnes)),
								minusOnes], minusOnes);
								
			for (var layer of discrepancer.layers) {
			if (layer.name.includes('sequential')) {
			  for (var lay of layer.layers) {
				if (! lay.name.includes('input')) {
				  lay.trainable = true;
				  if (constraint == true) {
				  lay.kernel.val.assign(lay.kernel.constraint.apply(lay.kernel.val));
				  lay.bias.val.assign(lay.bias.constraint.apply(lay.bias.val));
				  };
				  };
			  };
			};
			};
			
			if (constraint == true) {
				for (var layer of weightsPredictor.layers) {
					if (! layer.name.includes('input')) {
					  layer.kernel.val.assign(layer.kernel.constraint.apply(layer.kernel.val));
					  layer.bias.val.assign(layer.bias.constraint.apply(layer.bias.val));
					  };
				};
				for (var layer of task.layers) {
					if (! layer.name.includes('input')) {
					  layer.kernel.val.assign(layer.kernel.constraint.apply(layer.kernel.val));
					  layer.bias.val.assign(layer.bias.constraint.apply(layer.bias.val));
					  };
				  };
			};

			await discrepancer.trainOnBatch([xBatchSource, xBatchTarget,
								tf.neg(yBatchSource.reshapeAs(minusOnes)),
								tf.neg(yBatchTarget.reshapeAs(minusOnes)),
								sourceWeights, minusOnes], minusOnes);
			};
		};
		
		if (dataset == "Toy") {
			saveHistoryTask.push(task.predictOnBatch(xbatchTensor).as1D().dataSync());
			saveHistoryWeight.push(weightsPredictor.predictOnBatch(xsTensor).as1D().dataSync());
		};
		
		//wann.evaluate([xBatchSource, xBatchTarget, 
		//			 yBatchSource.reshapeAs(minusOnes),
		//			 yBatchTarget.reshapeAs(minusOnes),
		//			minusOnes], minusOnes).print();
	};
	
	bar.style.width = "100%";
	percentage.innerHTML = "100%";
	
	if (dataset == "Toy") {
	var copyHistory = [saveHistoryTask, saveHistoryWeight];
	localStorage.setItem("copyHistory", JSON.stringify(copyHistory));
	};
	return;
};


import * as tf from '@tensorflow/tfjs';
import Plotly from 'plotly.js-dist';
import 'bootstrap/dist/css/bootstrap.css';
import { linspace, srcLabelsk, extractXY, scale, mean, std, standardScaling, sum, coerceFloat, isFloat, jsonToArray } from './js/utils.js';
import {baseModelTwoLayer, baseModelOneLayer, baseModelZeroLayer} from './js/basic_methods.js';
import {createWann, trainWann} from './js/wann.js';
import fs from 'fs';
import "babel-polyfill";


// Var and func for Toy experiment
document.getElementById("viz").disabled = true;
document.getElementById("fitButton").disabled = true;
const xInput = linspace(-0.5, 0.5, 100);
const xInputPlot = linspace(0, 1, 100);
var xsPlot = [];
var xs = [];
var ys = [];
var yt = [];
var xtt = [];
var xttPlot;
var ytt = [];
var isFitting = false;
var stopTraining = false;
localStorage.setItem("stoptraining", stopTraining);
var copyHistory = [];
localStorage.setItem("copyHistory", copyHistory);
var xbatchInput = linspace(0, 1, 32);

function fitModelToy() {
	
	if (isFitting) {
		stopTraining = true;
		localStorage.setItem("stoptraining", stopTraining);
        setTimeout(fitModelToy, 50);
        return;
    } else {
		stopTraining = false;
		localStorage.setItem("stoptraining", stopTraining);
	};
	
	var epochs = parseInt(document.getElementById('epochs').value);
	var batchSize = parseInt(document.getElementById('batchSize').value);
	var lr = parseFloat(document.getElementById('lrToy').value);
	var method = document.getElementById('model-select').value;
		
	var xsTensor = tf.tensor2d(xs, [xs.length, 1]);
	var ysTensor = tf.tensor2d(ys, [ys.length, 1]);
	var xbatchTensor = tf.tensor2d(linspace(-0.5, 0.5, 32), [32, 1]);
	
	var xtTensor = tf.tensor2d(xInput, [xInput.length, 1]);
	var ytTensor = tf.tensor2d(yt, [yt.length, 1]);
	
	var xttTensor = tf.tensor2d(xtt, [xtt.length, 1]);
	var yttTensor = tf.tensor2d(ytt, [ytt.length, 1]);
	
	if (method == "noReweight"){
		var xTrainTensor = xsTensor.concat(xttTensor);
		var yTrainTensor = ysTensor.concat(yttTensor);
	};
	if (method == "srcOnly"){
		var xTrainTensor = xsTensor;
		var yTrainTensor = ysTensor;
	};
	if (method == "tgtOnly"){
		var xTrainTensor = xttTensor;
		var yTrainTensor = yttTensor;
	};
	
	var saveHistory = [];
	
	if (method == "wann")
	{
		
		var ySource = ysTensor.as1D();
		var yTarget = yttTensor.as1D();
		
		var shape = 1;
		const models = createWann(shape, baseModelTwoLayer, baseModelZeroLayer, undefined, undefined, "ones", lr)
		const wann = models[0];
		const weightsPredictor = models[1];
		const task = models[2];
		const discrepancer = models[3];
		
		isFitting = true;
		var startDate = new Date();
		trainWann(wann, weightsPredictor, task, discrepancer,
		epochs, batchSize, xsTensor, ySource, xttTensor, yTarget, "Toy", xbatchTensor, false).then(() => {
		isFitting = false;
		var endDate = new Date();
		document.getElementById("time").innerText = "Time \n" + ((endDate.getTime() - startDate.getTime()) / 1000).toFixed(1);
		var yPred = task.predict(xtTensor).as1D();
		var yTrue = ytTensor.as1D();
		var mse = tf.metrics.meanSquaredError(yTrue, yPred);
		document.getElementById("output").innerText = "MSE \n" + mse.dataSync()[0].toFixed(3);
		document.getElementById("viz").disabled = false;		
		});
		
	} 
	else 
	{
		const model = baseModelTwoLayer(undefined, lr, 1);
		
		var i = 1;
		var bar = document.getElementById('progress_bar');
		var percentage = document.getElementById('progress_percentage');
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
		
		function update(epoch) {
			model.predictOnBatch(xbatchTensor).reshape([32]).array().then(array => saveHistory.push(array));
			
			if (epoch > 0) {
				var diff = parseInt(epoch * ( 100 / epochs)) - parseInt((epoch-1) * ( 100 / epochs));
				for (var j=0 ; j < diff; j++) {
					move();
				};
			};
			
			if (epoch == (epochs-1)) {
				
				bar.style.width = "100%";
				percentage.innerHTML = "100%";
			};
			
			if (stopTraining) {
				model.stopTraining = true;
			};
		};
		
		move();
		
		isFitting = true;
		var startDate = new Date();
		model.fit(xTrainTensor, yTrainTensor, {epochs: epochs,
					   batchSize: batchSize,
					   callbacks: {onEpochEnd: (epoch, logs) => update(epoch)}
					   }).then(() => {
					   var endDate   = new Date();
					  document.getElementById("time").innerText = "Time \n" + ((endDate.getTime() - startDate.getTime()) / 1000).toFixed(1);
					   isFitting = false;
						//copyHistory = Array.from(saveHistory);
						localStorage.setItem("copyHistory", JSON.stringify(saveHistory));
						document.getElementById("viz").disabled = false;
						
						var yPred = tf.reshape(model.predict(xtTensor), [yt.length]);
						var yTrue = tf.reshape(ytTensor, [yt.length]);
						var mse = tf.metrics.meanSquaredError(yTrue, yPred);
						document.getElementById("output").innerText = "MSE \n" + mse.dataSync()[0].toFixed(3);

					   
					   });
	
	};
};

function animate() {
	
	
	
	var animation =	JSON.parse(localStorage.getItem("copyHistory"));
	
	if (animation.length == 2) {
	
	var pred = jsonToArray(animation[0]);
	var weight = jsonToArray(animation[1]);
	
	var aLength = pred.length; 
	for (var i = 0; i < aLength; i++) {
		
		var size = tf.tensor1d(weight[i]);
		var maxSize = tf.max(size).dataSync();
		size = size.dataSync();
		var sizeScaled = [];
		for (var w of size) {sizeScaled.push(20 * (w / maxSize));};
		
		Plotly.animate('toy',
			{data: [{x: xbatchInput, y: pred[i]}, {x: xsPlot, y: ys, marker:{size: sizeScaled}}]},
			{
			transition: {
			  duration: 0,
			},
			frame: {
			  duration: 0,
			  redraw: false,
			}
		})
	};
	} else {
	var aLength = animation.length; 
	for (var i = 0; i < aLength; i++) {
	
		Plotly.animate('toy',
			{data: [{x: xbatchInput, y: animation[i]}]},
			{
			transition: {
			  duration: 0,
			},
			frame: {
			  duration: 0,
			  redraw: false,
			}
		})
	};
	};
};



// Var and func for Kin experiment
document.getElementById("fitKinButton").disabled = true;
var xSrcTensorKin = tf.tensor2d([[0]], [1, 1]);
var ySrcTensorKin = tf.tensor2d([[0]], [1, 1]);
var xTgtTensorKin = tf.tensor2d([[0]], [1, 1]);
var yTgtTensorKin = tf.tensor2d([[0]], [1, 1]);
var xTrainTensorKin = tf.tensor2d([[0]], [1, 1]);
var yTrainTensorKin = tf.tensor2d([[0]], [1, 1]);
var xTestTensorKin = tf.tensor2d([[0]], [1, 1]);
var yTestTensorKin = tf.tensor2d([[0]], [1, 1]);
var isFittingKin = false;
var stopTrainingKin = false;

function fitModelKin() {
	
	if (isFittingKin) {
		stopTrainingKin = true;
		localStorage.setItem("stoptrainingKin", stopTrainingKin);
        setTimeout(fitModelKin, 50);
        return;
    } else {
		stopTrainingKin = false;
		localStorage.setItem("stoptrainingKin", stopTrainingKin);
	};
	
	var epochs = document.getElementById('epochsKin').value;
	var batchSize = document.getElementById('batchSizeKin').value;
	var projConst = parseFloat(document.getElementById('projConstKin').value);
	var projConstW = parseFloat(document.getElementById('projConstWKin').value);
	var lr = parseFloat(document.getElementById('lrKin').value);
	var method = document.getElementById('model-select-kin').value;
	
	if (method == "noReweightKin" || method == "wannKin"){
		var xTrain = xTrainTensorKin;
		var yTrain = yTrainTensorKin;
	};
	if (method == "srcOnlyKin"){
		var xTrain = xSrcTensorKin;
		var yTrain = ySrcTensorKin;
	};
	if (method == "tgtOnlyKin"){
		var xTrain = xTgtTensorKin;
		var yTrain = yTgtTensorKin;
	};
	
	if (method == "wannKin")
	{
		
		var xSource = xTrain.slice([0, 0], [ySrcTensorKin.shape[0], 8]);
		var ySource = yTrain.as1D().slice(0, ySrcTensorKin.shape[0]);
		
		var xTarget = xTrain.slice([ySrcTensorKin.shape[0], 0], [yTgtTensorKin.shape[0], 8]);
		var yTarget = yTrain.as1D().slice(ySrcTensorKin.shape[0], yTgtTensorKin.shape[0]);
		
		
		var shape = 8;
		const models = createWann(shape, baseModelOneLayer, baseModelOneLayer, projConst, projConstW, "zeros", lr)
		const wann = models[0];
		const weightsPredictor = models[1];
		const task = models[2];
		const discrepancer = models[3];
		
		isFittingKin = true;
		var startDate = new Date();
		trainWann(wann, weightsPredictor, task, discrepancer,
		epochs, batchSize, xSource, ySource, xTarget, yTarget, "Kin", undefined, true).then(() => {
		isFittingKin = false;
		var endDate = new Date();
		document.getElementById("timeKin").innerText = "Time \n" + ((endDate.getTime() - startDate.getTime()) / 1000).toFixed(1);
		var yPred = task.predict(xTestTensorKin).as1D();
		var yTrue = yTestTensorKin.as1D();
		var mse = tf.metrics.meanSquaredError(yTrue, yPred);
		document.getElementById("outputKin").innerText = "MSE \n" + mse.dataSync()[0].toFixed(4);
		
		yPred = yPred.dataSync();
		var yTestArray = yTestTensorKin.as1D().dataSync();
		var yTrainArray = yTrain.as1D().dataSync();
		var yPredTrain = task.predict(xTrain).as1D().dataSync();
		var yConcat = yTestTensorKin.as1D().concat(yTrain.as1D()).dataSync();
					
		validationPlot(yTrainArray, yPredTrain, yTestArray, yPred, yConcat, "Kin");
		
		
		});
		
	} 
	else 
	{
	
	
	const model = baseModelOneLayer(projConst, lr, 8);
	
	var i = 1;
	var bar = document.getElementById('progress_bar_kin');
	var percentage = document.getElementById('progress_percentage_kin');
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
	
	function update (epoch, logs) {		
		if (epoch > 0) {
			var diff = parseInt(epoch * ( 100 / epochs)) - parseInt((epoch-1) * ( 100 / epochs));
			for (var j=0 ; j < diff; j++) {
				move();
			};
		};
		
		if (epoch == (epochs-1)) {
			
			bar.style.width = "100%";
			percentage.innerHTML = "100%";
		};
		
		if (stopTrainingKin) {
			model.stopTraining = true;
		};
	};
	
	move();
	
	
	isFittingKin = true;
	var startDate = new Date();
	model.fit(xTrain, yTrain, {epochs: parseInt(epochs),
				   batchSize: parseInt(batchSize),
				   callbacks: {
					   onEpochEnd: (epoch, logs) => update(epoch, logs),
					   onBatchEnd: (epoch, logs) => {
						   for (var layer of model.layers) {
								layer.kernel.val.assign(layer.kernel.constraint.apply(layer.kernel.val));
								layer.bias.val.assign(layer.bias.constraint.apply(layer.bias.val));
							  };
					   }
				   }
				   }).then(() => {
				   var endDate   = new Date();
				  document.getElementById("timeKin").innerText = "Time \n" + ((endDate.getTime() - startDate.getTime()) / 1000).toFixed(1);
				   isFittingKin = false;
					
					var yPred = model.predict(xTestTensorKin).reshapeAs(yTestTensorKin);
					var mse = tf.metrics.meanSquaredError(yTestTensorKin, yPred);
					document.getElementById("outputKin").innerText = "MSE \n" + mse.dataSync()[0].toFixed(4);
					
					yPred = yPred.dataSync();
					var yTestArray = yTestTensorKin.as1D().dataSync();
					var yTrainArray = yTrain.as1D().dataSync();
					var yPredTrain = model.predict(xTrain).as1D().dataSync();
					
					var yConcat = yTestTensorKin.as1D().concat(yTrain.as1D()).dataSync();
					
					validationPlot(yTrainArray, yPredTrain, yTestArray, yPred, yConcat, "Kin");
					
				   
				   });

	}
}


function loadKin(domain) {
	if (domain == 'kin-8fh') {
		var string = fs.readFileSync(__dirname + '/../rsc/kin-8fh.csv', 'utf8').toString().replace(/\r\n/g,'\n').split('\n');
	};
	if (domain == 'kin-8fm') {
		var string = fs.readFileSync(__dirname + '/../rsc/kin-8fm.csv', 'utf8').toString().replace(/\r\n/g,'\n').split('\n');
	};
	if (domain == 'kin-8nm') {
		var string = fs.readFileSync(__dirname + '/../rsc/kin-8nm.csv', 'utf8').toString().replace(/\r\n/g,'\n').split('\n');
	};
	if (domain == 'kin-8nh') {
		var string = fs.readFileSync(__dirname + '/../rsc/kin-8nh.csv', 'utf8').toString().replace(/\r\n/g,'\n').split('\n');
	};
	return string;
};




// Var and func for UCI experiment
document.getElementById("fitUCIButton").disabled = true;
var xUCIsource = [];
var yUCIsource = [];
var xUCItargetTrain = [];
var yUCItargetTrain = [];
var xUCItargetTest = [];
var yUCItargetTest = [];
var xUCITrain = [];
var yUCITrain = [];
var isFittingUCI = false;
var stopTrainingUCI = false;


function fitModelUCI() {
	
	if (isFittingUCI) {
		stopTrainingUCI = true;
		localStorage.setItem("stoptrainingUCI", stopTrainingUCI);
        setTimeout(fitModelUCI, 50);
        return;
    } else {
		stopTrainingUCI = false;
		localStorage.setItem("stoptrainingUCI", stopTrainingUCI);
		};
	
	var epochs = parseInt(document.getElementById('epochsUCI').value);
	var batchSize = parseInt(document.getElementById('batchSizeUCI').value);
	var projConst = parseFloat(document.getElementById('projConstUCI').value);
	var projConstW = parseFloat(document.getElementById('projConstWUCI').value);
	var lr = parseFloat(document.getElementById('lrUCI').value);
	var method = document.getElementById('model-select-UCI').value;
		
	if (method == "noReweightUCI" || method == "wannUCI"){
		var xTrain = xUCITrain;
		var yTrain = yUCITrain;
	};
	if (method == "srcOnlyUCI"){
		var xTrain = xUCIsource;
		var yTrain = yUCIsource;
	};
	if (method == "tgtOnlyUCI"){
		var xTrain = xUCItargetTrain;
		var yTrain = yUCItargetTrain;
	};
	
	var output = standardScaling(xTrain, xUCItargetTest);
	xTrain = output[0];
	var xTest = output[1];
	var meanY = mean(yTrain);
	var stdY = std(yTrain, meanY);
	yTrain = scale(yTrain, meanY, stdY);
	var yTest = scale(yUCItargetTest, meanY, stdY);
	
	var xTrainTensor = tf.tensor2d(xTrain, [xTrain.length, 80], "float32");
	var yTrainTensor = tf.tensor2d(yTrain, [yTrain.length, 1], "float32");
	
	var xTestTensor = tf.tensor2d(xTest, [xTest.length, 80], "float32");
	var yTestTensor = tf.tensor1d(yTest, "float32");
	
	
	if (method == "wannUCI")
	{
		
		var xSource = xTrainTensor.slice([0, 0], [yUCIsource.length, 80]);
		var ySource = yTrainTensor.as1D().slice(0, yUCIsource.length);
		
		var xTarget = xTrainTensor.slice([yUCIsource.length, 0], [yUCItargetTrain.length, 80]);
		var yTarget = yTrainTensor.as1D().slice(yUCIsource.length, yUCItargetTrain.length);
		
		
		var shape = 80;
		const models = createWann(shape, baseModelOneLayer, baseModelOneLayer, projConst, projConstW, "zeros", lr)
		const wann = models[0];
		const weightsPredictor = models[1];
		const task = models[2];
		const discrepancer = models[3];
		
		isFittingUCI = true;
		var startDate = new Date();
		trainWann(wann, weightsPredictor, task, discrepancer,
		epochs, batchSize, xSource, ySource, xTarget, yTarget, "UCI", undefined, true).then(() => {
		isFittingUCI = false;
		var endDate = new Date();
		document.getElementById("timeUCI").innerText = "Time \n" + ((endDate.getTime() - startDate.getTime()) / 1000).toFixed(1);
		var yPred = task.predict(xTestTensor).as1D();
		var yTrue = yTestTensor.as1D();
		var mse = tf.metrics.meanSquaredError(yTrue, yPred);
		document.getElementById("outputUCI").innerText = "MSE \n" + mse.dataSync()[0].toFixed(4);
		
		yPred = yPred.dataSync();
		var yPredTrain = task.predict(xTrainTensor).as1D().dataSync();
		var yConcat = yTestTensor.as1D().concat(yTrainTensor.as1D()).dataSync();
					
		validationPlot(yTrain, yPredTrain, yTest, yPred, yConcat, "UCI");
		
		
		});
		
	} 
	else 
	{
		
	const model = baseModelOneLayer(projConst, lr, 80);
	
	var i = 1;
	var bar = document.getElementById('progress_bar_UCI');
	var percentage = document.getElementById('progress_percentage_UCI');
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
	
	function update (epoch, logs) {		
		if (epoch > 0) {
			var diff = parseInt(epoch * ( 100 / epochs)) - parseInt((epoch-1) * ( 100 / epochs));
			for (var j=0 ; j < diff; j++) {
				move();
			};
		};
		
		if (epoch == (epochs-1)) {
			
			bar.style.width = "100%";
			percentage.innerHTML = "100%";
		};
		
		if (stopTrainingUCI) {
			model.stopTraining = true;
		};
	};
	
	move();
		
	isFittingUCI = true;
	var startDate = new Date();
	model.fit(xTrainTensor, yTrainTensor, {epochs: parseInt(epochs),
				   batchSize: parseInt(batchSize),
				   callbacks: {
					   onEpochEnd: (epoch, logs) => update(epoch, logs),
					   onBatchEnd: (epoch, logs) => {
							   for (var layer of model.layers) {
									layer.kernel.val.assign(layer.kernel.constraint.apply(layer.kernel.val));
									layer.bias.val.assign(layer.bias.constraint.apply(layer.bias.val));
								  };
					   }
				   }
				   }).then(() => {
					   var endDate   = new Date();
					document.getElementById("timeUCI").innerText = "Time \n" + ((endDate.getTime() - startDate.getTime()) / 1000).toFixed(1);
				    isFittingUCI = false;
				
					
					var yPred = model.predict(xTestTensor).reshapeAs(yTestTensor);
					var mse = tf.metrics.meanSquaredError(yTestTensor, yPred);
					document.getElementById("outputUCI").innerText = "MSE \n" + mse.dataSync()[0].toFixed(4);
					
					yPred = yPred.dataSync();
					var yPredTrain = model.predict(xTrainTensor).reshape([yTrain.length]).dataSync();
					var yConcat = yTestTensor.as1D().concat(yTrainTensor.as1D()).dataSync();
					
					validationPlot(yTrain, yPredTrain, yTest, yPred, yConcat, "UCI");
				   });

	}
}


function validationPlot(yTrain, yPredTrain, yTest, yPred, yConcat, dataset){
	
	if (dataset == "UCI") {
		var divVal = document.getElementById("UCIval");
		var divHist = document.getElementById("UCIhist");
	};
	if (dataset == "Kin") {
		var divVal = document.getElementById("KinVal");
		var divHist = document.getElementById("KinHist");
	};
	
	divVal.style = "width:450px;height:450px;";
	divHist.style = "width:450px;height:450px;";
	
	var xMax = Math.max(...yConcat);
	var xMin = Math.min(...yConcat);

	var layoutVal = {
		  title: {
			text:'Validation Plot'
		  },
		  xaxis: {
			title: {
			  text: 'True'
			  }
			},
		  yaxis: {
			title: {
			  text: 'Predicted'
				}
			}
		};
		
	var layoutHist = {
		  title: {
			text:'Residuals'
		  },
		  xaxis: {
			title: {
			  text: 'Absolute Error'
			  }
			},
		  barmode: "overlay"
		};
	
	Plotly.newPlot(divVal, [{
		  x: yTrain,
		  y: yPredTrain,
		  mode: 'markers',
		  type: 'scatter',
		  name: 'train',
		  opacity: 0.6,
		}, 
		{
		  x: yTest,
		  y: yPred,
		  mode: 'markers',
		  type: 'scatter',
		  name: 'test',
		  opacity: 0.6,
		},
		{
		  x: [xMin, xMax],
		  y: [xMin, xMax],
		  mode: 'lines',
		  type: 'scatter',
		  name: 'test',
		  opacity: 0.6,
		  showlegend: false,
		  line:{color: 'black'}
		}
		
		], layoutVal);
		
		
		var errorTrain = [];
		for (var i=0; i<yTrain.length; i++) {
			errorTrain.push(yPredTrain[i] - yTrain[i]); 
		};
		var errorTest = [];
		for (var i=0; i<yTest.length; i++) {
			errorTest.push(yPred[i] - yTest[i]); 
		};
		
		Plotly.newPlot(divHist, [{
		  x: errorTrain,
		  type: "histogram", 
		  name: 'train',
		  opacity: 0.5,
		}, 
		{
		  x: errorTest,
		  type: "histogram", 
		  name: 'test',
		  opacity: 0.5,
		}], layoutHist);
	
	
	
};


function loadUCI(domain) {
	if (domain == 'low') {
		var string = fs.readFileSync(__dirname + '/../rsc/low.csv', 'utf8').toString().replace(/\r\n/g,'\n').split('\n');
	};
	if (domain == 'midle-low') {
		var string = fs.readFileSync(__dirname + '/../rsc/midle_low.csv', 'utf8').toString().replace(/\r\n/g,'\n').split('\n');
	};
	if (domain == 'midle-high') {
		var string = fs.readFileSync(__dirname + '/../rsc/midle_high.csv', 'utf8').toString().replace(/\r\n/g,'\n').split('\n');
	};
	if (domain == 'high') {
		var string = fs.readFileSync(__dirname + '/../rsc/high.csv', 'utf8').toString().replace(/\r\n/g,'\n').split('\n');
	};
	return string;
};

	
document.getElementById('loadtoyButton').addEventListener('click', (el, ev) => {
	xsPlot = [];
	xs = [];
	ys = [];
	
	var srcNbr = parseInt(document.getElementById('srcNbr').value)+1;
	var ampShift = parseFloat(document.getElementById('ampShift').value);
	var noiseLvl = parseFloat(document.getElementById('noiseLvl').value);
	var tgtLabels = parseInt(document.getElementById('tgtLabels').value);
	var tgtNoiseLvl = parseFloat(document.getElementById('tgtNoiseLvl').value);
	
	var count = 0;
	var data = [];
	var k = linspace(-1, 1, srcNbr);
	var kLength = k.length;
	var target_index = 1;//parseInt(Math.random() * kLength);
	for (var i = 0; i < kLength; i++) {
		
		if (i == target_index) {
			xtt = linspace(-0.5, 0.5, tgtLabels);
			xttPlot = linspace(0, 1, tgtLabels);
			yt = srcLabelsk(xInput, noiseLvl, ampShift, k[i]);
			ytt = srcLabelsk(xtt, tgtNoiseLvl, ampShift, k[i]);
			var plot_t = {
				  x: xInputPlot,
				  y: yt,
				  mode: 'lines',
				  type: 'scatter',
				  name: 'target',
				  line:{color: 'green', size:20}
				};
			var plot_tt = {
				  x: xttPlot,
				  y: ytt,
				  mode: 'markers',
				  type: 'scatter',
				  name: 'target train',
				  marker:{color: 'black', size: 10}
				};
			
			
		}
		
		else {
			var ysk = srcLabelsk(xInput, noiseLvl, ampShift, k[i]);
			var showLegend = false
			if (count == 0) {
				showLegend = true;
				count++;
			};
				
			var plot_k = {
				  x: xInputPlot,
				  y: ysk,
				  mode: 'lines',
				  type: 'scatter',
				  name: 'source',
				  showlegend: showLegend,
				  line:{color: 'blue'},
				};
			data.push(plot_k);
			xs = xs.concat(xInput);
			xsPlot = xsPlot.concat(xInputPlot);
			ys = ys.concat(ysk)
			};
		};
	data.push(plot_t);
	data.push(plot_tt);
	
		
	var div = document.getElementById("toy");
	div.style = "width:800px;height:400px;";
	
	Plotly.newPlot(div, data);
	document.getElementById("fitButton").disabled = false;
});



document.getElementById('fitButton').addEventListener('click', (el, ev) => {
	
	fitModelToy();
	
});


document.getElementById('viz').addEventListener('click', (el, ev) => {
	
	const traceSrc = {
		x: xsPlot,
		y: ys,
		mode: 'markers',
		type: 'scatter',
		name: 'source',
		opacity: 0.3,
		marker:{color: 'blue'}};
		
	const traceTgt = {
		x: xInputPlot,
		y: yt,
		mode: 'lines',
		type: 'scatter',
		name: 'target',
		opacity: 0.7,
		line:{color: 'green', size:20}};
	
	const traceTgtTrain = {
		x: xttPlot,
		y: ytt,
		mode: 'markers',
		type: 'scatter',
		name: 'target train',
		opacity: 0.7,
		marker:{color: 'black', size: 10}};

	Plotly.newPlot('toy', [{
		  x: xbatchInput,
		  y: tf.zeros([32]).dataSync(),
		  mode: 'lines',
		  type: 'scatter',
		  name: 'predict',
		  line:{color: 'red'}
		}, traceSrc, traceTgt, traceTgtTrain], {displayModeBar: false});
		
	animate();
});



document.getElementById('loadkinButton').addEventListener('click', (el, ev) => {
	
	var source = document.getElementById('kin-source').value;
	var target = document.getElementById('kin-target').value;
	var srcSize = parseInt(document.getElementById('srcKin').value);
	var tgtTrainSize = parseInt(document.getElementById('tgtTrainKin').value);
	var tgtTestSize = parseInt(document.getElementById('tgtTestKin').value);
	
	const sourceString = loadKin(source);
	const targetString = loadKin(target);
	
	var output_s = extractXY(sourceString, 8, srcSize);
	var xKinsource = output_s[0];
	var yKinsource = output_s[1];
	
	var output_tr = extractXY(targetString, 8, tgtTrainSize);
	var xKintargetTrain = output_tr[0];
	var yKintargetTrain = output_tr[1];
	
	var output_te = extractXY(targetString, 8, tgtTestSize);
	var xKintargetTest = output_te[0];
	var yKintargetTest = output_te[1];
	
	var xKinTrain = xKinsource.concat(xKintargetTrain);
	var yKinTrain = yKinsource.concat(yKintargetTrain);
	
	xTgtTensorKin = tf.tensor2d(xKintargetTrain, [xKintargetTrain.length, 8], "float32");
	yTgtTensorKin = tf.tensor2d(yKintargetTrain, [yKintargetTrain.length, 1], "float32");
	
	xSrcTensorKin = tf.tensor2d(xKinsource, [xKinsource.length, 8], "float32");
	ySrcTensorKin = tf.tensor2d(yKinsource, [yKinsource.length, 1], "float32");
	
	xTrainTensorKin = tf.tensor2d(xKinTrain, [xKinTrain.length, 8], "float32");
	yTrainTensorKin = tf.tensor2d(yKinTrain, [yKinTrain.length, 1], "float32");
	
	xTestTensorKin = tf.tensor2d(xKintargetTest, [xKintargetTest.length, 8], "float32");
	yTestTensorKin = tf.tensor1d(yKintargetTest, "float32");
	
	document.getElementById("fitKinButton").disabled = false;
});



document.getElementById('fitKinButton').addEventListener('click', (el, ev) => {
	
	fitModelKin();

});



document.getElementById('loadUCIButton').addEventListener('click', (el, ev) => {
		
	var source = document.getElementById('UCI-source').value;
	var target = document.getElementById('UCI-target').value;
	var srcSize = parseInt(document.getElementById('srcUCI').value);
	var tgtTrainSize = parseInt(document.getElementById('tgtTrainUCI').value);
	var tgtTestSize = parseInt(document.getElementById('tgtTestUCI').value);
	
	const sourceString = loadUCI(source);
	const targetString = loadUCI(target);
	
	var output_s = extractXY(sourceString, 80, srcSize);
	xUCIsource = coerceFloat(output_s[0]);
	yUCIsource = output_s[1].map(Number);
	
	var output_tr = extractXY(targetString, 80, tgtTrainSize);
	xUCItargetTrain = coerceFloat(output_tr[0]);
	yUCItargetTrain = output_tr[1].map(Number);
	
	var output_te = extractXY(targetString, 80, tgtTestSize);
	xUCItargetTest = coerceFloat(output_te[0]);
	yUCItargetTest = output_te[1].map(Number);
	
	xUCITrain = xUCIsource.concat(xUCItargetTrain);
	yUCITrain = yUCIsource.concat(yUCItargetTrain);
		
	document.getElementById("fitUCIButton").disabled = false;
});


function extractXYV2 (string, nCols, size) {
	var array = [];
	for (var i in string) {
		array.push(string[i].split(','));
	};
	var arrayFloat = Array.from(array);
	var x = arrayFloat.map(function(value,index) { return value.slice(0, nCols); });
	var y = arrayFloat.map(function(value,index) { return value[nCols];});
	return [x, y];
};



document.getElementById('fitUCIButton').addEventListener('click', (el, ev) => {
	
	fitModelUCI();

});	
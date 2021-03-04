// Utility functions

export function isFloat(n){
    return Number(n) === n;
}

export function jsonToArray(json) {
	var newArray = [];
	for (var key in json) {
		var row = [];
		for (var key2 in json[key]) {
			row.push(json[key][key2]);
		}
		newArray.push(row);
	}
	return newArray;
};



export function linspace(startValue, stopValue, cardinality) {
  var arr = [];
  var step = (stopValue - startValue) / (cardinality - 1);
  for (var i = 0; i < cardinality; i++) {
    arr.push(startValue + (step * i));
  }
  return arr;
};

export function srcLabelsk(arr, noise, ampShift, k) {
  var y = [];
  var arrLength = arr.length;
  for (var i = 0; i < arrLength; i++) {
	var arri = arr[i] + 0.5;
	var pow = Math.pow(arri, 3);
	var sin = Math.sin(10 * arri);
	var yij = k * ampShift * pow + sin + Math.sign(Math.random() - 0.5) * Math.random() * noise;
	y.push(yij);
	};
  return y;
};


function getRandomSubarray(arr, size) {
    //var arr = [...Array(length).keys()];
	var shuffled = arr.slice(0), i = arr.length, temp, index;
    while (i--) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
    }
    return shuffled.slice(0, size);
};


export function extractXY (string, nCols, size) {
	var array = [];
	for (var i in string) {
		array.push(string[i].split(','));
	};
	var arrayFloat = Array.from(array);
	//var nCols = arrayFloat[0].length - 1;
	var extract = getRandomSubarray(arrayFloat, size);
	var x = extract.map(function(value,index) { return value.slice(0, nCols); });
	var y = extract.map(function(value,index) { return value[nCols];});
	return [x, y];
};


export function sum (a) { return a.reduce(function(x,y) { return x + y; }) }
export function mean (a) { return sum(a) / a.length }
export function std (a, av) { return Math.sqrt(mean(a.map(function(x) { return (x - av) * (x - av); }))); }
export function scale (a, av, sd) { return a.map(function(x) { return (x - av) / sd}) }

export function standardScaling (array, arrayTest) {
	var newArray = [];
	var newArrayTest = [];
	arrayTest = arrayTest[0].map((col, i) => arrayTest.map(row => row[i]));
	array = array[0].map((col, i) => array.map(row => row[i]));
	for (var i=0; i<array.length; i++) {
		var av = mean(array[i]);
		var sd = std(array[i], av);
		var scaledLine = scale(array[i], av, sd);
		var scaledLineTest = scale(arrayTest[i], av, sd);
		newArray.push(scaledLine);
		newArrayTest.push(scaledLineTest);
	}
	newArray = newArray[0].map((col, i) => newArray.map(row => row[i]));
	newArrayTest = newArrayTest[0].map((col, i) => newArrayTest.map(row => row[i]));
	return [newArray, newArrayTest];
}


export function coerceFloat(a) {
	return a.map(function(elem) {
		return elem.map(function(elem2) {
			return parseFloat(elem2);
		});
	});
};


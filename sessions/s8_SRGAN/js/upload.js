function readURL(input) {
	if (input.files && input.files[0]) {
		var reader = new FileReader();

		reader.onload = function (e) {
			$('#uploadedImage')
				.attr('src', e.target.result)
				.width(150)
				.height(200);
		};

		reader.readAsDataURL(input.files[0]);
	}
}

function gan(url) {
	$.ajax({
		async: true,
		crossDomain: true,
		method: 'GET',
		url: url,
		processData: false,
		contentType: false
	})
	.done(function (response) {
        console.log(response);
		responseJson = response;
		if(responseJson.imagebytes){
			if(responseJson.imagebytes.length > 1){
				document.getElementById("ItemPreview").src = responseJson.imagebytes;
				document.getElementById('result').textContent = '';
			}
			else{
				document.getElementById('result').textContent = 'Got some junk Image !!!'
			}
		}
		else{
			document.getElementById('result').textContent = 'SORRY ;;;; Model didn\'t return anything !!!'
		}
	})
	.fail(function (error) {
		alert("There was an error while processing the model"); 
		console.log(error);
	});
    
}

function srgan(url){
	var fileInput = document.getElementById('resnet34FileUpload').files;
	if(!fileInput.length){
		return alert('Please choose a file to upload first');
	}
	
	var file = fileInput[0];
	var filename = file.name;
	
	var formData = new FormData();
	formData.append(filename, file);
	
	console.log(filename);
	console.log(url);
	console.log('Processing...');

	document.getElementById('result').textContent = 'Processing...';
    $.ajax({
		async: true,
		crossDomain: true,
		method: 'POST',
		url: url,
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})
	.done(function (response) {
		responseJson = JSON.parse(response);
        console.log(responseJson);
        classes = {0 :'Jay Panda', 1: 'Jyoti Basu',2: 'Manoj Bajpayee', 3: 'Menaka Gandhi', 4: 'Naveen Patnaik', 5: 'Nitish Kumar', 6: 'Om Puri', 7:'Pragyan Ojha', 8: 'Sambit Patra', 9: 'Sushant Singh'};
		if(responseJson.imagebytes){
			if(responseJson.imagebytes.length > 1){
				document.getElementById("ItemPreview").src = responseJson.imagebytes;
				document.getElementById('result').textContent = '';
			}
			else{
				document.getElementById('result').textContent = 'Image does not have any face, Pls upload image containing face !!!'
			}
		}
		else if(responseJson.predicted >= 0 && responseJson.predicted <= 9){
			document.getElementById('result').textContent = classes[responseJson.predicted];
		}
		else{
			document.getElementById('result').textContent = 'SORRY ;;;; May be your model doesnt predict anything !!!'
		}
	})
	.fail(function (error) {
		alert("There was an error while processing the model"); 
		console.log(error);
	});
};


function uploadAndClassifyImage(url){
	var fileInput = document.getElementById('resnet34FileUpload').files;
	if(!fileInput.length){
		return alert('Please choose a file to upload first');
	}
	
	var file = fileInput[0];
	var filename = file.name;
	
	var formData = new FormData();
	formData.append(filename, file);
	
	console.log(filename);
	console.log(url);
	console.log('Processing...');

	document.getElementById('result').textContent = 'Processing...';
    $.ajax({
		async: true,
		crossDomain: true,
		method: 'POST',
		url: url,
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})
	.done(function (response) {
		responseJson = JSON.parse(response);
        console.log(responseJson);
        classes = {0 :'Jay Panda', 1: 'Jyoti Basu',2: 'Manoj Bajpayee', 3: 'Menaka Gandhi', 4: 'Naveen Patnaik', 5: 'Nitish Kumar', 6: 'Om Puri', 7:'Pragyan Ojha', 8: 'Sambit Patra', 9: 'Sushant Singh'};
		if(responseJson.imagebytes){
			if(responseJson.imagebytes.length > 1){
				document.getElementById("ItemPreview").src = responseJson.imagebytes;
				document.getElementById('result').textContent = '';
			}
			else{
				document.getElementById('result').textContent = 'Image does not have any face, Pls upload image containing face !!!'
			}
		}
		else if(responseJson.predicted >= 0 && responseJson.predicted <= 9){
			document.getElementById('result').textContent = classes[responseJson.predicted];
		}
		else{
			document.getElementById('result').textContent = 'SORRY ;;;; May be your model doesnt predict anything !!!'
		}
	})
	.fail(function (error) {
		alert("There was an error while processing the model"); 
		console.log(error);
	});
};



//$('#btnResNetUpload').click(uploadAndClassifyImage);
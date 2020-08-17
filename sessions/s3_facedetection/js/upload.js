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
		if(responseJson.imagebytes){
			if(responseJson.imagebytes.length > 1){
				document.getElementById("ItemPreview").src = responseJson.imagebytes;
				document.getElementById('result').textContent = '';
			}
			else{
				document.getElementById('result').textContent = 'Image does not have any face, Pls upload image containing face !!!'
			}
		}
		else if(responseJson.predicted >= 0){
			document.getElementById('result').textContent = responseJson.predicted;
		}
		else{
			document.getElementById('result').textContent = 'Image does not have any face, Pls upload image containing face !!!'
		}
	})
	.fail(function (error) {
		alert("There was an error while sending prediction request to resnet34 model."); 
		console.log(error);
	});
};

//$('#btnResNetUpload').click(uploadAndClassifyImage);
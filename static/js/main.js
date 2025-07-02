// static/js/main.js
$(document).ready(function () {
    // Initial setup
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Image preview function
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('.image-section').fadeIn(500);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('#result').hide();
        readURL(this);
        $('#btn-predict').show();
    });

    // Update confidence bar function
    function updateConfidenceBar(confidenceText) {
        const percentage = parseFloat(confidenceText.replace('%', ''));
        if (!isNaN(percentage)) {
            $('.confidence-bar-fill').css('width', percentage + '%');
            
            // Remove all confidence classes first
            $('.confidence-bar-fill').removeClass('confidence-low confidence-medium confidence-high');
            
            // Add appropriate class based on confidence level
            if (percentage > 90) {
                $('.confidence-bar-fill').addClass('confidence-high');
            } else if (percentage > 70) {
                $('.confidence-bar-fill').addClass('confidence-medium');
            } else {
                $('.confidence-bar-fill').addClass('confidence-low');
            }
        }
    }

    // Prediction handling
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        
        $('.loader').show();
        $(this).hide();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                $('.loader').hide();
                if (data.prediction) {
                    let resultText = data.prediction;
                    if (data.confidence) {
                        resultText += `<br><span class="confidence-score">Confidence: ${data.confidence}</span>`;
                        updateConfidenceBar(data.confidence);
                    }
                    $('#result').fadeIn(600).find('p').html(resultText);
                } else if (data.error) {
                    $('#result').fadeIn(600).find('p').text(data.error);
                    $('.confidence-bar-fill').css('width', '0%');
                }
            },
            error: function (xhr, status, error) {
                $('.loader').hide();
                $('#result').fadeIn(600).find('p').text('Error: Could not process prediction');
                $('#btn-predict').show();
            }
        });
    });
    
    // Adjust image preview height on window resize
    $(window).resize(function() {
        $('.img-preview').height($('.img-preview').width()); // Maintain square aspect ratio
    }).resize(); // Trigger resize on load
});

// --------------------------------------------------------
// // static/js/main.js
// $(document).ready(function () { // Waits for the HTML document to be fully loaded before running the code inside
//     // Initial setup
//     $('.image-section').hide(); // Hides the image preview section initially
//     $('.loader').hide(); // Hides the loading spinner initially
//     $('#result').hide(); // Hides the result display section initially

//     // Image preview function
//     function readURL(input) { // Defines a function to handle image preview when a file is selected
//         if (input.files && input.files[0]) { // Checks if there are files and at least one file is selected
//             var reader = new FileReader(); // Creates a new FileReader object to read file contents
//             reader.onload = function (e) { // Defines what happens when the file is successfully read
//                 $('#imagePreview').css('background-image', 'url(' + e.target.result + ')'); // Sets the preview div's background to the uploaded image
//                 $('.image-section').fadeIn(500); // Shows the image preview section with a 500ms fade-in effect
//             }
//             reader.readAsDataURL(input.files[0]); // Reads the file as a data URL (base64 encoded string)
//         }
//     }

//     $("#imageUpload").change(function () { // Triggers when a new file is selected in the file input
//         $('#result').hide(); // Hides any previous result
//         readURL(this); // Calls the readURL function with the current input element
//         $('#btn-predict').show(); // Shows the predict button
//     });

//     // Prediction handling
//     $('#btn-predict').click(function () { // Triggers when the predict button is clicked
//         var form_data = new FormData($('#upload-file')[0]); // Creates a FormData object from the form for file upload
        
//         $('.loader').show(); // Shows the loading spinner
//         $(this).hide(); // Hides the predict button during processing

//         $.ajax({ // Initiates an AJAX request to the server
//             type: 'POST', // Specifies the HTTP method as POST
//             url: '/predict', // Sets the endpoint URL for prediction from app.py
//             data: form_data, // Attaches the form data (image file) to the request
//             contentType: false, // Prevents jQuery from setting a content type (needed for file upload)
//             cache: false, // Disables caching of the request
//             processData: false, // Prevents jQuery from processing the data (needed for file upload)
//             success: function (data) { // Callback function for successful response
//                 $('.loader').hide(); // Hides the loading spinner
//                 if (data.prediction) { // Checks if the response contains a prediction
//                     $('#result').fadeIn(600).find('p').text(data.prediction); // Shows the result with prediction text and a 600ms fade-in
//                 } else if (data.error) { // Checks if the response contains an error
//                     $('#result').fadeIn(600).find('p').text(data.error); // Shows the result with error text and a 600ms fade-in
//                 }
//             },
//             error: function (xhr, status, error) { // Callback function for failed request
//                 $('.loader').hide(); // Hides the loading spinner
//                 $('#result').fadeIn(600).find('p').text('Error: Could not process prediction'); // Shows a generic error message
//                 $('#btn-predict').show(); // Shows the predict button again for retry
//             }
//         });
//     });
    
//     // Adjust image preview height on window resize
//     $(window).resize(function() {
//         $('.img-preview').height($('.img-preview').width()); // Maintain square aspect ratio
//     }).resize(); // Trigger resize on load
// });

/* 
    Here, AJAX:

    Sends the uploaded image to the /predict endpoint asynchronously.
    Shows a loader while waiting for the response.
    Updates only the #result section with the prediction or error message when the server responds, without reloading the page.
*/

/*
    AJAX, which stands for Asynchronous JavaScript and XML, is a set of web development techniques that allows a web page to communicate with a server and update parts of itself without requiring a full page reload. 
    Here's a breakdown of what AJAX does in general:

    Core Purpose
    AJAX enables asynchronous communication between the client (browser) and the server. 
    This means that instead of the traditional approach—where submitting a form or clicking a link causes the entire page to refresh—AJAX lets you send and receive data in the background, 
    updating only specific parts of the page dynamically.

    How It Works
        1- JavaScript Triggers a Request:
            - Typically, an event (like a button click or form submission) triggers JavaScript code to initiate an AJAX request.
            - In modern implementations, this is often done using the XMLHttpRequest object or the newer fetch API.
        
        2- Asynchronous Communication:
            - The request is sent to the server without interrupting the user's interaction with the page. The browser doesn't freeze or reload; it continues to function normally.
            - "Asynchronous" means the JavaScript code doesn't wait for the server's response before moving on—it handles the response later via a callback function or promise.

        3- Server Processes the Request:
            - The server (e.g., a Flask app in Python) receives the request, processes it (e.g., runs a prediction model), and sends back a response.
            - The response can be in various formats, such as JSON, XML, HTML, or plain text (JSON is most common today).
            
        4- Dynamic Page Update:
            - Once the response is received, JavaScript updates the webpage dynamically by manipulating the DOM (Document Object Model).
            - For example, it might update a <div> with new text, show an image, or display a result—all without refreshing the entire page.
*/
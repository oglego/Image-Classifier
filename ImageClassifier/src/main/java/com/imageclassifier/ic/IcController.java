package com.imageclassifier.ic;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * Controller class to handle image classification requests.
 *
 * This class defines the endpoint for classifying images and provides
 * a REST API for users to upload an image file. The classification logic
 * is delegated to the IcService class.
 */

@RestController
@RequestMapping("/api")
public class IcController {

    @Autowired
    private IcService imageClassifierService; // Service for handling image classification logic

    /**
     * Endpoint for classifying the uploaded image.
     *
     * This method accepts an image file as input, processes it through
     * the IcService, and returns the classification result.
     *
     * @param file The image file to be classified.
     * @return ResponseEntity containing the classification result or error message.
     */
    @PostMapping("/classify")
    public ResponseEntity<?> classifyImage(@RequestParam("file") MultipartFile file) {
        // Classify the image using the IcService and return the result
        try {
            String result = imageClassifierService.classifyImage(file);
            return ResponseEntity.ok(result); // Return a successful response with the result
        } catch (Exception e) {
            // Return an error response if classification fails
            return ResponseEntity.badRequest().body("Error: " + e.getMessage());
        }
    }
}

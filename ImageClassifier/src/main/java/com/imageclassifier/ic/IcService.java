package com.imageclassifier.ic;

import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.multipart.MultipartFile;

/**
 * Service class to handle image classification requests.
 *
 * This service sends the uploaded image to an external API for classification
 * and retrieves the classification result. The WebClient is used to perform
 * the HTTP request to the classification API.
 */
@Service
public class IcService {

    private final WebClient webClient; // WebClient used to communicate with the classification API

    /**
     * Constructor for IcService.
     *
     * Initializes the WebClient with a base URL pointing to the classification service.
     *
     * @param webClientBuilder The builder for constructing WebClient instances.
     */
    public IcService(WebClient.Builder webClientBuilder) {
        // Set the base URL for the classification API
        this.webClient = webClientBuilder.baseUrl("http://0.0.0.0:8000").build();
    }

    /**
     * Classifies the uploaded image by sending it to an external classification API.
     *
     * This method creates a multipart request to send the image file to the
     * API and retrieves the classification result as a string.
     *
     * @param file The image file to be classified.
     * @return The classification result as a string.
     */
    public String classifyImage(MultipartFile file) {
        // Build the multipart body with the image file
        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        bodyBuilder.part("file", file.getResource()); // Add the file as a part of the multipart request

        // Send the image to the classification API and retrieve the response
        return this.webClient.post() // Perform a POST request
                .uri("/classify") // The endpoint for classification
                .contentType(MediaType.MULTIPART_FORM_DATA) // Set the content type for multipart data
                .bodyValue(bodyBuilder.build()) // Add the body with the image
                .retrieve() // Execute the request
                .bodyToMono(String.class) // Extract the response body as a string
                .block();  // Block until the response is received
    }
}

package com.imageclassifier.ic;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Main entry point for the Image Classifier Spring Boot application.
 *
 * This class contains the main method to launch the Spring Boot application
 * and start the embedded web server. The @SpringBootApplication annotation
 * is used to enable automatic configuration, component scanning, and
 * configuration properties.
 */
@SpringBootApplication
public class IcApplication {
	/**
     * Main method to start the Spring Boot application.
     *
     * @param args Command-line arguments (if any)
     */
	public static void main(String[] args) {
		SpringApplication.run(IcApplication.class, args); // Launch the application
	}

}

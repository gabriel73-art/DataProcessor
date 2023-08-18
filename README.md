# DataProcessor
The Data Processor is a Java tool that allows processing an input file with records in different formats and performing specific searches on people's data. 
The processing is done in batches and in parallel to improve performance.

## Prerequisites
Java 8 or higher installed on the system.
There is an input.txt file in the root folder

## Usage
To execute the Data Processor, follow the following command-line syntax:
java -jar data-processor.jar input.txt COMMAND ARGUMENT

Where:

input.txt is the input file containing the records to be processed.
COMMAND is the search command to perform. It can be "CITY" to search people by city or "ID" to search people by identification.
ARGUMENT is the value to be used for the search. For example, the name of a city or a specific id.

## Input File Format
The input file must contain records in different formats. Each line of the file represents a record with the following format: F or D. 
Records starting with "F" indicate a new data format and are used to group people's data under that format.
Records starting with "D" contain information about a person under the previously defined format.

Format and data examples:

Formato F1:

F1

D Peter Parker,NEW YORK,12345k

Formato F2:

F2

D Peter Parker ; NEW YORK ; 12345-k

## Results
The Data Processor will display a list of results in the console based on the provided command and argument. 
If the search yields no results, it will display the message "No matching data found."

## Implementation
The Data Processor uses batch processing and parallelism to improve efficiency. 
Each batch of records is processed independently, and the results are later combined to obtain the final response.

## Example Usage
Suppose we have an input file named "data.txt" with records of people in different formats. 
To search for all people belonging to the city "Barcelona," the following command would be used:
data.txt
CITY
Barcelona
The Data Processor will process the file and display in the console the names and identifications of all people who have been in "Barcelona" . 
If no matches are found, it will display the message "No matching data found."

package org.example;

import org.example.POJO.Person;
import org.example.Utils.ProcessBatchTask;
import org.example.Utils.UtilsMethod;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

public class DataProcessor {
    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    private static final int BATCH_SIZE = 8;

    public static void main(String[] args) {
       /* Scanner scanner = new Scanner(System.in);

        System.out.print("Ingrese el nombre del archivo de entrada: ");
        String inputFile = scanner.nextLine();

        System.out.print("Ingrese el comando (CITY o ID): ");
        String command = scanner.nextLine();

        System.out.print("Ingrese ciudad o id: ");
        String argument = scanner.nextLine();*/

        String inputFile = args[0];


        String command = args[1];


        String argument = String.join(" ", Arrays.copyOfRange(args, 2, args.length));

        try (BufferedReader reader = new BufferedReader(new FileReader(inputFile))) {
            ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
            List<Future<Map<String, List<Person>>>> futures = new ArrayList<>();
            String line;
            List<String> batchLines = new ArrayList<>();
            List<String> searchResults = new ArrayList<>();

            while ((line = reader.readLine()) != null) {
                batchLines.add(line);

                if (batchLines.size() >= BATCH_SIZE) {
                    List<String> batch = new ArrayList<>(batchLines);
                    Future<Map<String, List<Person>>> future = executor.submit(new ProcessBatchTask(batch));
                    futures.add(future);
                    batchLines.clear();
                }
            }

            for (Future<Map<String, List<Person>>> future : futures) {
                try {
                    Map<String, List<Person>> batchData = future.get();
                    List<String> answers;
                    if (command.equalsIgnoreCase("CITY")) {
                        answers = UtilsMethod.getPersonByCity(batchData, argument);
                    } else if (command.equalsIgnoreCase("ID")) {
                        answers = UtilsMethod.getCitiesById(batchData, argument);
                    } else {
                        answers = new ArrayList<>();
                    }
                    searchResults.addAll(answers);
                } catch ( ExecutionException e) {
                    System.out.println("Error processing batch: " + e.getMessage());
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            executor.shutdown();
            UtilsMethod.printSTDOUT(searchResults);
        } catch (IOException e) {
            System.out.println("Error reading the input file: " + e.getMessage());
        }
    }

}

package org.example.Utils;

import org.example.POJO.Person;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class  UtilsMethod {

    public static List<String> getPersonByCity(Map<String, List<Person>> batchData, String city) {
        List<String> personInCity = new ArrayList<>();
        List<Person> people = batchData.get("F1");
        List<Person> people2 = batchData.get("F2");
        for (Person person : people) {
            if(person.getCity().equalsIgnoreCase(city))
                personInCity.add(person.getName()+","+person.getId());
        }
        for (Person person : people2) {
            if(person.getCity().equalsIgnoreCase(city))
                personInCity.add(person.getName()+","+person.getId());
        }
        return personInCity;
    }

   /* public static List<String> getCitiesById(Map<String, List<Person>> batchData , String id) {
        List<String> personInCity = new ArrayList<>();
        List<Person> people = batchData.get("F1");
        List<Person> people2 = batchData.get("F2");

        for (Person person : people) {
            if(person.getId().equals(id.contains("-")? id.replace("-", ""): id))
                personInCity.add(person.getCity());
        }
        for (Person person : people2) {
            if(person.getId().equals(id.contains("-")? id: id.substring(0, id.length() - 1) + "-" + id.charAt(id.length() - 1)))
                personInCity.add(person.getCity());
        }
        return personInCity;
    }*/
   public static List<String> getCitiesById(Map<String, List<Person>> batchData, String id) {
       List<String> personInCity = new ArrayList<>();
       ExecutorService executor = Executors.newFixedThreadPool(2); // Dos hilos

       Future<List<String>> future1 = executor.submit(() -> getCitiesByIdInBatch(batchData, "F1", id));
       Future<List<String>> future2 = executor.submit(() -> getCitiesByIdInBatch(batchData, "F2", id));

       try {
           personInCity.addAll(future1.get());
           personInCity.addAll(future2.get());
       } catch (InterruptedException | ExecutionException e) {
           e.printStackTrace();
       }

       executor.shutdown();
       return personInCity;
   }

    public static void printSTDOUT(List<String> searchResults) {
        if (searchResults.isEmpty()) {
            System.out.println("No matching data found.");
        } else {
            for (String city : searchResults) {
                System.out.println(city);
            }
        }
    }

    private static List<String> getCitiesByIdInBatch(Map<String, List<Person>> batchData, String batchKey, String id) {
        List<String> personInCity = new ArrayList<>();
        List<Person> people = batchData.get(batchKey);

        if(batchKey.equals("F1"))
        for (Person person : people) {
            if (person.getId().equals(id.contains("-") ? id.replace("-", "") : id))
                personInCity.add(person.getCity());
        }
        if(batchKey.equals("F2"))
            for (Person person : people) {
                if(person.getId().equals(id.contains("-")? id: id.substring(0, id.length() - 1) + "-" + id.charAt(id.length() - 1)))
                    personInCity.add(person.getCity());
            }

        return personInCity;
    }

}

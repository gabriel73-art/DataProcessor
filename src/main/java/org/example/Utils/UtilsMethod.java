package org.example.Utils;

import org.example.POJO.Person;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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

    public static List<String> getCitiesById(Map<String, List<Person>> batchData , String id) {
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

}

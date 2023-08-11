package org.example.Utils;

import org.example.POJO.Person;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

public class ProcessBatchTask implements Callable<Map<String, List<Person>>> {
    private final List<String> batchLines;

    public ProcessBatchTask(List<String> batchLines) {
        this.batchLines = batchLines;
    }

    @Override
    public Map<String, List<Person>> call() {
        Map<String, List<Person>> batchData = new HashMap<>();
        batchData.put("F1", new ArrayList<>());
        batchData.put("F2", new ArrayList<>());
        String currentFormat = "";

        for (String line : batchLines) {
            if (line.startsWith("F1")) {
                currentFormat = "F1";
            } else if (line.startsWith("F2")) {
                currentFormat = "F2";
            } else if (line.startsWith("D")) {
                if(currentFormat.isEmpty())
                    currentFormat = line.contains(",")? "F1": "F2";
                if (!batchData.isEmpty()) {
                    List<Person> people = batchData.get(currentFormat);
                    Person person = parsePerson(line.substring(2), currentFormat);
                    if (person != null) {
                        people.add(person);
                    }
                }
            }
        }

        return batchData;
    }



    private Person parsePerson(String line, String format) {


        if (format.equals("F1") ) {
            String name = line.split(",")[0].trim();
            String city = line.split(",")[1].trim();
            String id = line.split(",")[2].trim();
            return new Person(name, city, id);
        } else if (format.equals("F2")) {
            String[] f2Fields = line.split(";");
            if (f2Fields.length == 3) {
                String name = f2Fields[0].trim();
                String city = f2Fields[1].trim();
                String id = f2Fields[2].trim();

                return new Person(name, city, id);
            }
        }
        return null;
    }
}



package macro;

import java.io.*;
import java.util.*;
import star.common.*;
import star.base.neo.*;
import java.text.DecimalFormat;

public class RunTestCaseMacroV5 extends StarMacro {

    public void execute() {
        Simulation sim = getActiveSimulation();

        // Extract test case number from simulation name
        String simName = sim.getPresentationName();
        int caseNum = -1;
        try {
            String[] parts = simName.split("_");
            for (int i = 0; i < parts.length; i++) {
                if (parts[i].equalsIgnoreCase("TestCase")) {
                    caseNum = Integer.parseInt(parts[i+1]);
                    break;
                }
            }
        } catch (Exception e) {
            sim.println("Failed to extract case number from sim name: " + simName);
            return;
        }

        if (caseNum == -1) {
            sim.println("Could not determine case number from simulation name.");
            return;
        }

        sim.println("Running Test Case #" + caseNum);
        String csvPath = "/mnt/data1/test_cases.csv";

        List<String> headers = new ArrayList<>();
        List<String> values = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String line = br.readLine();
            if (line != null) {
                headers = Arrays.asList(line.split(","));
            }

            int currentLine = 0;
            while ((line = br.readLine()) != null) {
                currentLine++;
                if (currentLine == caseNum) {
                    values = Arrays.asList(line.split(","));
                    break;
                }
            }

            if (values.isEmpty()) {
                sim.println("Test case number " + caseNum + " not found in CSV.");
                return;
            }

        } catch (IOException e) {
            sim.println("Failed to read CSV: " + e.getMessage());
            return;
        }

        // Apply parameters
        for (int i = 0; i < headers.size(); i++) {
            String paramName = headers.get(i).trim();
            String paramValue = values.get(i).trim();

            try {
                double val = Double.parseDouble(paramValue);
                ScalarGlobalParameter param =
                    (ScalarGlobalParameter) sim.get(GlobalParameterManager.class).getObject(paramName);
                param.getQuantity().setValue(val);
                sim.println("Set " + paramName + " = " + val);
            } catch (Exception e) {
                sim.println("WARNING: Could not set parameter " + paramName + ": " + e.getMessage());
            }
        }

        // Run simulation
        long t0 = System.currentTimeMillis();
        sim.println("Simulation start time (ms): " + t0);
        
        sim.getSimulationIterator().run();
        
        long t1 = System.currentTimeMillis();
        double elapsedSec = (t1 - t0) / 1000.0;
        DecimalFormat df = new DecimalFormat("#0.00");
        sim.println("Simulation end time   (ms): " + t1);
        sim.println("Simulation wall-time    : " + df.format(elapsedSec) + " s");

        // Define file paths
        String folderPath = "/mnt/data1/Test_Case_" + caseNum;
        new File(folderPath).mkdirs();
        String csvExportPath = folderPath + "/All_Data_Case_" + caseNum + ".csv";
        String simFilePath = folderPath + "/Test_Case_" + caseNum + ".sim";

        // Try extracting and exporting table
        try {
            Table table = sim.getTableManager().getTable("All_Data");
            table.extract();
            table.export(csvExportPath);
            sim.println("Exported table to " + csvExportPath);
        } catch (Exception e) {
            sim.println("WARNING: Table export failed or table 'All_Data' not found: " + e.getMessage());
        }

        // Save simulation to correct location
        try {
            sim.saveState(simFilePath);
            sim.println("Saved simulation to " + simFilePath);
        } catch (Exception e) {
            sim.println("WARNING: Failed to save simulation file: " + e.getMessage());
        }

        // Attempt to delete the saved .sim file
        File simFile = new File(simFilePath);
        if (simFile.exists() && simFile.delete()) {
            sim.println("Deleted simulation file: " + simFilePath);
        } else {
            sim.println("Failed to delete simulation file: " + simFilePath);
        }
    }
}

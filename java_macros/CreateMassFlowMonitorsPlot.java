package macro;

import star.common.*;
import star.base.neo.*;
import star.vis.*;
import star.flow.*;
import star.common.graph.*;
import java.util.*;
import star.base.report.*;
import star.common.graph.DataSet;


public class CreateMassFlowMonitorsPlot extends StarMacro {
    public void execute() {
        Simulation sim = getActiveSimulation();

        // Get the ReportManager
        ReportManager reportManager = sim.getReportManager();

        // Create a MonitorPlot
        MonitorPlot monitorPlot = sim.getPlotManager().createMonitorPlot();
        monitorPlot.setPresentationName("Mass Flow Plot");

        // Loop through all Reports
        for (Report report : reportManager.getObjects()) {
            if (report instanceof MassFlowReport) {
                String name = report.getPresentationName();
                if (name.startsWith("MFR_FA") && (name.endsWith("_Above") || name.endsWith("_Below"))) {

                    // Create Monitor
                    Monitor monitor = ((MassFlowReport) report).createMonitor();
                    monitor.setPresentationName("Monitor_" + name);

                    // Add Monitor to the plot
                    DataSet dataset = monitorPlot.getDataSet(monitor);
                    monitorPlot.getDataSetManager().addDataSet(dataset);
                }
            }
        }

        monitorPlot.open();
    }
}

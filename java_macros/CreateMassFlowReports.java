package macro;

import java.io.File;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.Collection;

import star.common.Simulation;
import star.common.Report;
import star.common.ReportManager;
import star.vis.DataSet;
import star.vis.MonitorPlot;
import star.vis.PlotExport;

public class CreateMassFlowMonitorsPlot extends StarMacro {

  @Override
  public void execute() {
    Simulation sim = getActiveSimulation();
    
    // 1) Build the monitor plot as before
    ReportManager rm = sim.getReportManager();
    MonitorPlot plot = sim.getPlotManager().createMonitorPlot();
    plot.setPresentationName("Mass Flow Plot");
    for (Report r : rm.getObjects()) {
      if (r instanceof star.common.MassFlowReport) {
        String nm = r.getPresentationName();
        if ( nm.startsWith("MFR_FA") && 
             (nm.endsWith("_Above") || nm.endsWith("_Below")) ) {
          star.common.Monitor m = ((star.common.MassFlowReport) r).createMonitor();
          m.setPresentationName("Monitor_" + nm);
          DataSet ds = plot.getDataSet(m);
          plot.getDataSetManager().addDataSet(ds);
        }
      }
    }
    plot.open();
    sim.println("Monitor plot created with " 
                + plot.getDataSetManager().getObjects().size() 
                + " series.");

    // Get the directory where this .sim is running
    String sessionDir = sim.getSessionDir();

    // 2) Export plot image
    try {
      PlotExport plotExport = new PlotExport(sim);
      String imgPath = sessionDir + File.separator + "MassFlowPlot.png";
      plotExport.encode(plot, imgPath, 1920, 1080);
      plotExport.close();
      sim.println("Saved plot image to: " + imgPath);
    } catch (Exception e) {
      sim.println("Failed to export plot image: " + e.getMessage());
    }

    // 3) Export each DataSet to CSV
    Collection<DataSet> dataSets = plot.getDataSetManager().getObjects();
    for (DataSet ds : dataSets) {
      String dsName = ds.getPresentationName().replaceAll("[^a-zA-Z0-9_]", "_");
      String csvPath = sessionDir + File.separator + dsName + ".csv";
      try (PrintWriter pw = new PrintWriter(new File(csvPath))) {
        // header
        pw.println("Time,Value");
        double[] x = ds.getXData();
        double[] y = ds.getYData();
        for (int i = 0; i < x.length; i++) {
          pw.println(x[i] + "," + y[i]);
        }
        sim.println("✔ Wrote dataset '" + dsName + "' to: " + csvPath);
      } catch (IOException ioe) {
        sim.println("Failed to write dataset '" 
                    + dsName + "': " + ioe.getMessage());
      }
    }
  }
}

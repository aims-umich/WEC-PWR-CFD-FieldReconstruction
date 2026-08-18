import java.util.*;
import star.common.*;

public class ListPartIdentifiersMacro extends StarMacro {
    public void execute() {
        Simulation sim = getActiveSimulation();
        Collection<Part> parts = sim.getPartManager().getObjects();
        for (Part part : parts) {
            sim.println("▶ " + part.toString());
        }
    }
}

package macro;

import java.util.*;
import star.common.*;
import star.vis.*;
import star.base.neo.*;

public class SetAbovePlanesOriginMacro extends StarMacro {

    public void execute() {
        Simulation sim = getActiveSimulation();
        // Grab all PlaneSection parts (including your FAxxx_Above derived planes)
        Collection<PlaneSection> planes = sim.getPartManager().getObjectsOf(PlaneSection.class);

        for (PlaneSection plane : planes) {
            String name = plane.getPresentationName();
            // Only target those ending in "_Above"
            if (name.endsWith("_Above")) {
                // Set origin to [0,2,0]
                plane.setOrigin(new DoubleVector(new double[] {0.0, 2.0, 0.0}));
            }
        }
    }
}

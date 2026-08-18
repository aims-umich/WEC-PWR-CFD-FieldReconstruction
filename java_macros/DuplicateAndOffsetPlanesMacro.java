
package macro;

import java.util.*;
import star.common.*;
import star.vis.*;
import star.base.neo.*;

public class DuplicateAndOffsetPlanesMacro extends StarMacro {

    public void execute() {
        Simulation sim = getActiveSimulation();
        Collection<PlaneSection> planes = sim.getPartManager().getObjectsOf(PlaneSection.class);

        for (PlaneSection originalPlane : planes) {
            String originalName = originalPlane.getPresentationName();

            // Rename original plane
            originalPlane.setPresentationName(originalName.replace("_Plane", "_Below"));

            // Create copy of plane
            PlaneSection copiedPlane = (PlaneSection) sim.getPartManager().createDerivedPart(PlaneSection.class);
            copiedPlane.setPresentationName(originalName.replace("_Plane", "_Above"));

            // Set input parts from original
            copiedPlane.setInputPartsInput(originalPlane.getInputPartsInput());

            // Offset origin by (0,1,0)
            DoubleVector origOrigin = originalPlane.getOrigin();
            double x = origOrigin.getComponent(0);
            double y = origOrigin.getComponent(1) + 1.0;
            double z = origOrigin.getComponent(2);
            copiedPlane.setOrigin(new DoubleVector(x, y, z));

            // Preserve orientation
            copiedPlane.setOrientation(originalPlane.getOrientation());
        }
    }
}

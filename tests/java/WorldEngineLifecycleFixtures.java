import java.util.List;
import java.util.Locale;
import org.frcworldstate.core.Geometry.*;
import org.frcworldstate.core.World.*;
import org.frcworldstate.core.WorldEngine;
import org.frcworldstate.core.PoseHistory;

/** Emits actual WorldEngine snapshots; synthetic observations/config, no robot I/O. */
public final class WorldEngineLifecycleFixtures {
    static final FieldIdentity FIELD = new FieldIdentity("OFFSEASON_2026", "synthetic-field-v1", "geometry-v1");
    static final long MEASUREMENT_US = 970_000;
    static final long EXPIRY_US = 700_000;

    static WorldEngine engine() {
        return new WorldEngine(new WorldEngine.Config(8, 16, 3, 500_000, EXPIRY_US,
            20_000, 30_000, .8, .5, .25, .01, 4, 2, .02, 10, .05, 3, .2, .01, false),
            new PoseHistory(20, 2_000_000), FIELD);
    }
    static EgoState ego(long time, boolean valid) {
        return new EgoState(time, new Pose2(new Vec2(1, 2), 0), Velocity2.zero(),
            new Uncertainty(.01, 0, .01), valid);
    }
    static ObservationFrame frame(long sequence, long capture, long publication) {
        return new ObservationFrame(0, new SourceStamp("camera-front", "camera-boot-1", "calibration-v1",
            "mount-v1", "same-capture", sequence, capture, publication, 0),
            List.of(new Observation("detection-" + sequence, "piece", .9, new Vec2(2, 0),
                new Uncertainty(.02, 0, .02))));
    }
    static String vec(Vec2 value) { return "{\"x\":" + value.x() + ",\"y\":" + value.y() + "}"; }
    static String uncertainty(Uncertainty value) {
        return "{\"xx_m2\":" + value.xxM2() + ",\"xy_m2\":" + value.xyM2() + ",\"yy_m2\":" + value.yyM2() + "}";
    }
    static String stamp(SourceStamp s) {
        return "{\"source_id\":\"" + s.sourceId() + "\",\"boot_id\":\"" + s.bootId()
            + "\",\"calibration_revision\":\"" + s.calibrationRevision() + "\",\"mount_revision\":\"" + s.mountRevision()
            + "\",\"correlation_group\":\"" + s.correlationGroup() + "\",\"sequence\":" + s.sequence()
            + ",\"capture_us\":" + s.captureUs() + ",\"publication_us\":" + s.publicationUs()
            + ",\"target_height_m\":" + s.targetHeightM() + "}";
    }
    static String snapshot(WorldSnapshot s) {
        EgoState e = s.ego();
        StringBuilder out = new StringBuilder("{\"id\":" + s.id() + ",\"epoch\":" + s.epoch()
            + ",\"obstacle_map_version\":" + s.obstacleMapVersion()
            + ",\"field\":{\"season\":\"" + s.field().season() + "\",\"map_id\":\"" + s.field().mapId()
            + "\",\"geometry_revision\":\"" + s.field().geometryRevision() + "\"},\"ego\":{\"time_us\":" + e.timeUs()
            + ",\"field_pose\":{\"position_m\":" + vec(e.fieldPose().position()) + ",\"heading_rad\":" + e.fieldPose().headingRad()
            + "},\"field_velocity\":{\"linear_mps\":" + vec(e.fieldVelocity().linearMps())
            + ",\"angular_radps\":" + e.fieldVelocity().angularRadPerSec() + "},\"uncertainty\":" + uncertainty(e.uncertainty())
            + ",\"localization_valid\":" + e.localizationValid() + "},\"tracks\":[");
        boolean comma = false;
        for (ObjectTrack t : s.tracks()) {
            if (comma) out.append(','); comma = true;
            out.append("{\"id\":" + t.id() + ",\"epoch\":" + t.epoch() + ",\"object_class\":\"" + t.objectClass()
                + "\",\"position_m\":" + vec(t.positionM()) + ",\"velocity_mps\":" + vec(t.velocityMps())
                + ",\"estimate_us\":" + t.estimateUs() + ",\"last_measurement_us\":" + t.lastMeasurementUs()
                + ",\"confidence\":" + t.confidence() + ",\"uncertainty\":" + uncertainty(t.uncertainty())
                + ",\"lifecycle\":\"" + t.lifecycle() + "\",\"estimate_kind\":\"" + t.kind() + "\",\"provenance\":[");
            for (int i = 0; i < t.provenance().size(); i++) {
                if (i > 0) out.append(','); out.append(stamp(t.provenance().get(i)));
            }
            out.append("]}");
        }
        return out.append("]}").toString();
    }
    static String scenario(String name, long publishedUs, boolean twoMeasurements, boolean valid, long receiveDelayUs) {
        WorldEngine engine = engine();
        engine.addEgo(ego(900_000, valid));
        String first = "null";
        if (twoMeasurements) {
            var decision = engine.accept(frame(1, 900_000, 900_000), 900_000);
            first = "{\"status\":\"" + decision.status() + "\",\"accepted_measurements\":" + decision.acceptedMeasurements() + "}";
        }
        engine.addEgo(ego(MEASUREMENT_US, valid));
        if (receiveDelayUs > 0) engine.addEgo(ego(MEASUREMENT_US + receiveDelayUs, valid));
        var decision = engine.accept(frame(2, MEASUREMENT_US, MEASUREMENT_US + receiveDelayUs), MEASUREMENT_US + receiveDelayUs);
        if (publishedUs > MEASUREMENT_US + receiveDelayUs) engine.addEgo(ego(publishedUs, valid));
        return "{\"name\":\"" + name + "\",\"capture_us\":" + MEASUREMENT_US + ",\"receive_us\":" + (MEASUREMENT_US + receiveDelayUs)
            + ",\"first_frame\":" + first + ",\"last_frame\":{\"status\":\"" + decision.status()
            + "\",\"accepted_measurements\":" + decision.acceptedMeasurements() + "},\"snapshot\":"
            + snapshot(engine.publish(ego(publishedUs, valid))) + "}";
    }
    public static void main(String[] args) {
        Locale.setDefault(Locale.ROOT);
        System.out.println("{\"engine_expiry_us\":" + EXPIRY_US + ",\"decision_max_track_age_us\":250000,\"scenarios\":["
            + scenario("confirmed_at_capture", MEASUREMENT_US, true, true, 0) + ","
            + scenario("delayed_camera_frame", 1_000_000, true, true, 25_000) + ","
            + scenario("fresh_coasting_after_dropout", 1_200_000, true, true, 25_000) + ","
            + scenario("decision_age_boundary", 1_210_000, true, true, 25_000) + ","
            + scenario("decision_age_expired", 1_210_001, true, true, 25_000) + ","
            + scenario("engine_expired", MEASUREMENT_US + EXPIRY_US + 1, true, true, 25_000) + ","
            + scenario("tentative_delayed", 1_000_000, false, true, 25_000) + ","
            + scenario("invalid_localization", 1_000_000, true, false, 25_000) + "]}");
    }
}

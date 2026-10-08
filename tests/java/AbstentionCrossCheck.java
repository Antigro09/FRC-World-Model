import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import org.frcworldstate.core.PredictionContractTest;
import org.frcworldstate.core.PredictionGate;
import static org.frcworldstate.core.Predictor.*;

/** Java emits the same explicit abstention expected from the Python binding. */
public final class AbstentionCrossCheck {
    public static void main(String[] args) throws Exception {
        var r=PredictionContractTest.fixtureRequest();
        var forecasts=new ArrayList<Forecast>();
        for(var candidate:r.candidates()) for(var target:r.targets()) {
            var samples=new ArrayList<ForecastSample>();
            for(long t=0;t<=r.horizonUs();t+=r.stepUs()) samples.add(new ForecastSample(t,false,0,0,0,0,0,0,0,null));
            forecasts.add(new Forecast(candidate.candidateId(),target.targetId(),samples));
        }
        var reply=new Reply(r.schemaVersion(),r.modelId(),r.requestId(),r.snapshotId(),r.epoch(),r.field(),r.source(),r.frame(),r.timestampDomain(),r.historyCutoffUs(),r.issuedUs(),r.validUntilUs(),r.horizonUs(),r.stepUs(),ForecastKind.LEARNED_FORECAST,"uncalibrated-abstention-v1",forecasts);
        var decision=PredictionContractTest.fixtureGate().evaluate(r,reply,r.history().get(1).snapshot(),r.issuedUs());
        if(decision.status()!=PredictionGate.Status.UNCALIBRATED || !decision.usesBaseline()) throw new AssertionError("masked uncalibrated reply changed baseline");
        Files.writeString(Path.of(args[0]),PredictionContractTest.replyJson(reply));
        System.out.println("Java abstention uses baseline: "+decision.status());
    }
}

ls
import Simulation as s

testSim = s.Simulation("config_tip.json")
testSim.setInteractionSphere()
#testSim.setInteractionTip()
testSim.setInteractionHalfPlane()
testSim.setInteractionLinearElastic()
testSim.setInteractionPointZeroForce()
#testSim.setInteractionCOMForce()
#testSim.setInteractionVelocityVerlet()
testSim.setInteractionOverDamped()
testSim.run()

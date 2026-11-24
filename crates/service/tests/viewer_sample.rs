//! The UR5 sample the viewer ships, imported with its real collision meshes
//! the way the page does it, must stand clear of the demo cell and pick both
//! parts from the ready posture the page sets.

use base64::Engine as _;
use serde_json::{json, Value};

/// The posture `UR5_READY` in `apps/viewer/script.part.js`: elbow up, tool
/// pointing down about 0.2 m above the cube.
const READY: [f64; 8] = [-0.49, -1.45, 0.64, -0.76, -1.57, -2.06, 0.0, 0.0];

fn call(session: &mut axisforge_service::Session, request: &Value) -> Value {
    serde_json::from_str(&session.handle_json(&request.to_string())).unwrap()
}

fn import_sample() -> axisforge_service::Session {
    let folder =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/viewer/samples/ur5");
    let urdf = std::fs::read_to_string(folder.join("robot.urdf")).unwrap();
    let mut files = serde_json::Map::new();
    for entry in std::fs::read_dir(folder.join("meshes")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        let bytes = std::fs::read(&path).unwrap();
        files.insert(
            name,
            base64::engine::general_purpose::STANDARD
                .encode(bytes)
                .into(),
        );
    }
    assert_eq!(files.len(), 7, "the sample ships seven collision meshes");
    let mut session = axisforge_service::Session::demo().unwrap();
    let reply = call(
        &mut session,
        &json!({
            "op": "import_robot", "format": "urdf", "text": urdf, "files": files,
            "attachGripper": "", "intoDemoCell": true,
        }),
    );
    assert_eq!(reply["ok"], true, "{reply}");
    let reply = call(
        &mut session,
        &json!({ "op": "set_joints", "values": READY }),
    );
    assert_eq!(reply["ok"], true, "{reply}");
    session
}

#[test]
fn the_ready_posture_touches_nothing() {
    let mut session = import_sample();
    let reply = call(&mut session, &json!({ "op": "snapshot", "values": READY }));
    let collision = &reply["data"]["collision"];
    assert_eq!(collision["conclusive"], true, "{collision}");
    // `base_link` hangs under a `world` link by a fixed joint; its mesh sits
    // on the floor and must not be reported as touching it.
    assert_eq!(collision["contacts"], json!([]), "{collision}");
    // Every moving link has its collision mesh, so that check covers the arm.
    let described = call(&mut session, &json!({ "op": "describe" }));
    assert_eq!(described["data"]["linksWithoutCollision"], json!([]));
}

#[test]
fn both_parts_can_be_picked_from_the_ready_posture() {
    let mut session = import_sample();
    for part in ["cube", "dowel"] {
        let id = session
            .project()
            .environment
            .body_by_name(part)
            .unwrap()
            .id
            .clone();
        let reply = call(&mut session, &json!({ "op": "plan_pick", "bodyId": id }));
        assert_eq!(reply["ok"], true, "{reply}");
        assert!(
            !reply["data"]["plans"].as_array().unwrap().is_empty(),
            "no plan for the {part}; rejections: {}",
            reply["data"]["rejections"]
        );
    }
}

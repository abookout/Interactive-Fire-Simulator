using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

// This camera controller script was taken from another project of mine that just happened to work beautifully for capturing video for this project.
public class CameraController : MonoBehaviour
{
    bool rClickHeld = false;
    Vector2 mouseDelta;
    Vector3 angularVelocity;

    float scrollDelta;
    float zoomVelocity;
    float zoomFactor;
    float zoomTarget;

    [SerializeField] Camera cam;

    [Header("Rotation parameters")]
    [SerializeField] float smoothTime;
    [SerializeField] float mouseRotateSpeed;

    [Header("Zoom parameters")]
    [SerializeField, Min(0)] float initialZoom = 15;
    [SerializeField, Min(0)] float minZoom;
    [SerializeField, Min(0)] float maxZoom;
    [SerializeField] float zoomSmoothTime;
    [SerializeField] float zoomSpeed;

    private void Start()
    {
        zoomFactor = initialZoom;
        zoomTarget = initialZoom;
        cam = Camera.main;
    }

    void Update()
    {
        if (!Application.isFocused)
        {
            scrollDelta = 0;
            return;
        }

        rClickHeld = Input.GetMouseButton(1);
        if (rClickHeld)
        {
            mouseDelta += new Vector2(Input.GetAxisRaw("Mouse X"), Input.GetAxisRaw("Mouse Y"));
            Cursor.lockState = CursorLockMode.Locked;
        }
        else
        {
            Cursor.lockState = CursorLockMode.None;
        }

        scrollDelta -= Input.GetAxis("Mouse ScrollWheel");
    }

    void FixedUpdate()
    {
        Vector3 moveDelta = new Vector3(mouseDelta.y, -mouseDelta.x, 0);
        Vector3 targetRot = transform.eulerAngles + mouseRotateSpeed * moveDelta;
        Vector3 newRot = Vector3.SmoothDamp(transform.eulerAngles, targetRot, ref angularVelocity, smoothTime);
        // Set x angle in worldspace and y in local space
        //transform.eulerAngles = new Vector3(transform.eulerAngles.x, newRot.y, transform.eulerAngles.z);
        //transform.localEulerAngles = new Vector3(newRot.x, transform.localEulerAngles.y, transform.localEulerAngles.z);

        zoomTarget += zoomSpeed * scrollDelta;
        zoomTarget = Mathf.Clamp(zoomTarget, minZoom, maxZoom);

        zoomFactor = Mathf.SmoothDamp(zoomFactor, zoomTarget, ref zoomVelocity, zoomSmoothTime);
        //cam.transform.localPosition = new Vector3(0, 0, -zoomFactor);

        transform.SetLocalPositionAndRotation(Vector3.zero - transform.forward * zoomFactor, Quaternion.Euler(newRot));

        mouseDelta = Vector2.zero;
        scrollDelta = 0;
    }
}

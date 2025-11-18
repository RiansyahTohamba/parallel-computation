import dpctl.tensor as dpt

a = dpt.arange(10, dtype=dpt.float32)
b = dpt.arange(10, dtype=dpt.float32) * 2

c = a + b  # operasi dilakukan di GPU Intel (SYCL)

print(c)
print("Device:", c.device)

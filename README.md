# CUDA BASIC EXAMPLES and GPU Parallel Programming  
  
สวัสดีครับ Examples นี้จัดทำขึ้นเพื่ออยากจะให้ทุกคนได้ลองเรียนรู้ และทำความเข้าใจเกี่ยวกับเรื่อง GPU Parallel Programming.  
ซึ่งโค้ด Examples ทั้งหมดเป็นแค่ตัวอย่างง่ายๆ เพื่อให้ทุกคนนำเอาไปใช้ทดสอบการทำงานของโปรแกรม และแก้ไขลองผิดลองถูกต่างๆ เพื่อให้เกิดความเข้าใจมากยิ่งขึ้น  
  
ใน Example สุดท้าย ซึ่งก็คือ Ex4.cu จะเป็น Example เดียวที่ใช้ประโยชน์จาก GPU Parallel Programming ซึ่งเป็นตัวอย่างการหาค่า min, max, และผลรวม จากชุดตัวเลขทั้งหมด 507,510,784 จำนวน  
ซึ่งใช้ประโยชน์จาก GPU ทั้ง Block และ Thread ในการประมวลผล ซึ่งผมได้ทำเวลาเปรียบเทียบไว้ให้ ระหว่าง GPU และ CPU หรือแม้กระทั่งการ Optimize loop ในตัวอย่าง CPU ด้วย (บางที Loop อาจจะไม่ได้เร็วเสมอไป...)
  
จากนั้นถ้าใครอยากจะลองฝึกแก้โจทย์ที่ผมจะให้ต่อจากนี้ เพื่อเป็นการยืนยันความเข้าใจของคุณก็ให้ทำการ Clone project ออกไปได้เลย  
  
คุณสมบัติ - สิ่งที่คุณอาจจะต้องมีพื้นฐานบ้างเล็กน้อย ได้แก่  
- ความรู้พื้นฐานเกี่ยวกับภาษา C/C++ ถ้าใครยังไม่มีแนะนำให้ไปฝึกกันก่อนสักนิด  
- ความรู้เกี่ยวกับการจัดการ Memory และ Pointer
- ความรู้พื้นฐานเกี่ยวกับเรื่อง Synchronous, Asynchronous, Concurrent, Parallel  
(จากที่ผมลองค้นหาเรื่องเหล่านี้ใน Google เป็นภาษาไทย พบว่ามีบทความอธิบายไว้ไม่ถูกต้องหลายบทความ ซึ่งอาจจะทำให้คนที่มาอ่านบทความเกิดความเข้าใจที่อาจจะคลาดเคลื่อนได้)  
- มีความสามรถในการแก้ปัญหาได้ดี  
- มีความอดทนพยายาม และกล้าที่จะลองผิดลองถูก  
  
ก่อนอื่นเลยหลังจากที่คุณคิดว่าคุณมีคุณสมบัติตามที่ผมเขียนไว้ด้านบนแล้วให้คุณลองศึกษา พื้นฐาน CUDA จากเองสารฉบับนี้ https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf  
โดยคุณจะต้อง Install สิ่งที่ต้องใช้เองทั้งหมด ทางผมจะไม่ได้สอนวิธีติดตั้งไว้ ถือว่าเป็นด่านแรกในการเรียนรู้  
  
ภาระกิจที่คิดจะต้องทำในภาระกิจนี้ก็คือ  
  
ภาระกิจหลัก  
1. ในไฟล์ Ex3.cu มีตัวอย่างต่างๆ เกี่ยวกับการทำงานของ Block, Thread ว่าแต่ละตัวทำงานแตกต่างกันอย่างไร  
(ซึ่งมันจะทำให้ทุกคนเช้าใจการทำงานแบบ Parallel มากยิ่งขึ้น แล้วเข้าใจความแตกต่างจริงๆ ของ Synchronous and Asynchronous)  
แต่ว่าใน Algorithm ที่ใช้ทั้ง Block and Thread ในการสร้างสูตรคูณ แม่ 2 ถึง แม้ 13 ยังมีปัญหาบางอย่างเกี่ยวกับลำดับการทำงานแบบ Parallel อยู่ สิ่งที่คุณต้องทำก็คือไปแก้ไขลำดับการทำงานให้ถูกต้อง  
สามารถดูตัวอย่างการทำงานของ Block and Thread แบบอย่างใดอย่างหนึ่งจากตัวอย่างใน function multiplicationTableBlock และ multiplicationTableThread เพื่อความเข้าใจมากยิ่งขึ้นได้...  
<img src="https://github.com/bagidea/cuda_basic_examles/blob/master/0.png" width="720">  
<img src="https://github.com/bagidea/cuda_basic_examles/blob/master/1.png" width="720">  
<img src="https://github.com/bagidea/cuda_basic_examles/blob/master/2.png" width="720">  
  
ภาระกิจเสริม  
1. สามารถอธิบายหลักการ GPU Parallel Programming ได้ชัดเจน  
2. สามารถอธิบายความแตกต่างระหว่าง Synchronous, Asynchronous, Concurrent, Parallel ได้ชัดเจน   
3. นำความรู้ที่ได้ไปเขียนเป็นบทความ และบอกต่อให้กับคนที่สนใจ...  
  
ถ้าทำกันได้แล้วอย่าลืมเอาไปใช้งานจริงในงานของพวกคุณด้วยนะครับ ^__^ ขอบคุณมากครับ Thnak you.  
  
#คนหนึ่งชีวิต  
#BagIdea  

<p><b>Please type in the following text:</b>

<?php

for ($i = 1; $i <=2000000; $i++) {

$md5 = md5(microtime() * mktime());


$captcha_string = substr($md5,0,5);

$name = $i."_" .$captcha_string.".jpg";
$captcha_img = imagecreatetruecolor(70, 40);

$color = imagecolorallocate($captcha_img, 255, 0, 255);

$line = imagecolorallocate($captcha_img,233,239,239);

imagestring($captcha_img, 5, 10, 10, $captcha_string, $color);
imageline($captcha_img,rand(0,10),rand(0,20),rand(10,50),rand(10,40),$line);
imageline($captcha_img,rand(0,10),rand(0,20),rand(10,60),rand(10,50),$line);
imageline($captcha_img,rand(0,10),rand(0,40),rand(10,70),rand(10,70),$line);

imagejpeg($captcha_img, "/home/geetika/captcha/dataset_ssd_1T/new_dataset_website/" . $name,100);
imagedestroy($captcha_img);


$_SESSION['key'] = md5($captcha_string);

}
?>



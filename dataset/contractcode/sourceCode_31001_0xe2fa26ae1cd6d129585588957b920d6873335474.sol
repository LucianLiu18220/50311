/**
 *Submitted for verification at Etherscan.io on 2016-04-08
*/

contract echo {
  /* Constructor */
  function () {
    msg.sender.send(msg.value);
  }
}